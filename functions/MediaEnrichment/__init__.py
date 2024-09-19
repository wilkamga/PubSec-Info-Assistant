import json
import logging
import os
import hashlib
from typing import Optional

import azure.functions as func
import requests
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient, exceptions
from shared_code.status_log import State, StatusClassification, StatusLog
from shared_code.utilities import Utilities, MediaType

from vi_search.language_models.azure_openai import OpenAI
from vi_search.prompt_content_db.azure_search import AzureVectorSearch
from vi_search.prep_scenes import get_sections_generator
from vi_search.language_models.language_models import LanguageModels
from vi_search.prompt_content_db.prompt_content_db import PromptContentDB, VECTOR_FIELD_NAME
from vi_search.vi_client.video_indexer_client import init_video_indexer_client, VideoIndexerClient

from azure.core.credentials import AzureKeyCredential
from datetime import datetime


azure_blob_storage_account = os.environ["BLOB_STORAGE_ACCOUNT"]
azure_blob_drop_storage_container = os.environ[
    "BLOB_STORAGE_ACCOUNT_UPLOAD_CONTAINER_NAME"
]
azure_blob_content_storage_container = os.environ[
    "BLOB_STORAGE_ACCOUNT_OUTPUT_CONTAINER_NAME"
]
azure_blob_storage_endpoint = os.environ["BLOB_STORAGE_ACCOUNT_ENDPOINT"]
azure_blob_storage_key = os.environ["AZURE_BLOB_STORAGE_KEY"]
azure_blob_connection_string = os.environ["BLOB_CONNECTION_STRING"]
azure_blob_content_storage_container = os.environ[
    "BLOB_STORAGE_ACCOUNT_OUTPUT_CONTAINER_NAME"
]
azure_blob_content_storage_container = os.environ[
    "BLOB_STORAGE_ACCOUNT_OUTPUT_CONTAINER_NAME"
]

enrichmentEndpoint = os.environ["ENRICHMENT_ENDPOINT"]

# Cosmos DB
cosmosdb_url = os.environ["COSMOSDB_URL"]
cosmosdb_key = os.environ["COSMOSDB_KEY"]
cosmosdb_log_database_name = os.environ["COSMOSDB_LOG_DATABASE_NAME"]
cosmosdb_log_container_name = os.environ["COSMOSDB_LOG_CONTAINER_NAME"]
cosmosdb_media_database_name = "mediadb"
cosmosdb_media_container_name = "mediahashes"

# Cognitive Services
cognitive_services_key = os.environ["ENRICHMENT_KEY"]
cognitive_services_endpoint = os.environ["ENRICHMENT_ENDPOINT"]
cognitive_services_account_location = os.environ["ENRICHMENT_LOCATION"]

# Search Service
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "gptkbindex"
SEARCH_CREDS = AzureKeyCredential(os.environ.get("AZURE_SEARCH_SERVICE_KEY"))

# Video Indexer Service
viAccountName =  os.environ["VI_ACCOUNT_NAME"]
azResourceGroup =  os.environ["RESOURCE_GROUP_NAME"]
azSubscriptionId = os.environ["SUBSCRIPTION_ID"]

# Translation params for OCR'd text
targetTranslationLanguage = os.environ["TARGET_TRANSLATION_LANGUAGE"]

API_DETECT_ENDPOINT = (
    f"{enrichmentEndpoint}language/:analyze-text?api-version=2023-04-01"
)
API_TRANSLATE_ENDPOINT = (
    f"{enrichmentEndpoint}translator/text/v3.0/translate?api-version=3.0"
)

MAX_CHARS_FOR_DETECTION = 1000
translator_api_headers = {
    "Ocp-Apim-Subscription-Key": cognitive_services_key,
    "Content-type": "application/json",
}


FUNCTION_NAME = "MediaEnrichment"

utilities = Utilities(
    azure_blob_storage_account=azure_blob_storage_account,
    azure_blob_storage_endpoint=azure_blob_storage_endpoint,
    azure_blob_drop_storage_container=azure_blob_drop_storage_container,
    azure_blob_content_storage_container=azure_blob_content_storage_container,
    azure_blob_storage_key=azure_blob_storage_key
)

# Initialize CosmosClient
cosmos_client = CosmosClient(url=cosmosdb_url, credential=cosmosdb_key)
# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(azure_blob_connection_string)

def generate_video_hash(blob_client):
    """Generate MD5 hash for the video file."""
    hash_md5 = hashlib.md5()
    stream = blob_client.download_blob().readall()
    hash_md5.update(stream)
    return hash_md5.hexdigest()

def check_video_exists(video_hash):
    """Check if the video hash already exists in the Cosmos DB."""
    container = cosmos_client.get_database_client(cosmosdb_media_database_name).get_container_client(cosmosdb_media_container_name)
    try:
        item = container.read_item(item=video_hash, partition_key=video_hash)
        return True
    except exceptions.CosmosResourceNotFoundError:
        return False

def upload_video_hash(video_id, file_name, video_hash):
    """Upload the video hash and video ID to the Cosmos DB."""
    container = cosmos_client.get_database_client(cosmosdb_media_database_name).get_container_client(cosmosdb_media_container_name)
    container.upsert_item({
        'id': video_hash,
        'video_id': video_id,
        'file_name': file_name
    })

def get_existing_video_id(video_hash):
    """Get the existing video ID from the Cosmos DB."""
    container = cosmos_client.get_database_client(cosmosdb_media_database_name).get_container_client(cosmosdb_media_container_name)
    item = container.read_item(item=video_hash, partition_key=video_hash)
    return item['video_id']


def main(msg: func.QueueMessage) -> None:
    """This function is triggered by a message in the media-enrichment-queue.
    It will first analyse the media. If the media is a video, it will then
    extract insights. """

    message_body = msg.get_body().decode("utf-8")
    message_json = json.loads(message_body)
    blob_path = message_json["blob_name"]
    blob_uri = message_json["blob_uri"]
    try:
        statusLog = StatusLog(
            cosmosdb_url, cosmosdb_key, cosmosdb_log_database_name, cosmosdb_log_container_name
        )
        
        logging.info(
            "Python queue trigger function processed a queue item: %s",
            msg.get_body().decode("utf-8"),
        )
        # Receive message from the queue
        statusLog.upsert_document(
            blob_path,
            f"{FUNCTION_NAME} - Received message from media-enrichment-queue ",
            StatusClassification.DEBUG,
            State.PROCESSING,
        )

        # Run the image through the Video Indexer service
        file_name, file_extension, file_directory = utilities.get_filename_and_extension(
            blob_path)
        blob_path_plus_sas = utilities.get_blob_and_sas(blob_path)

        data = {"url": f"{blob_path_plus_sas}"}

        blob_name = blob_path.split("/", 1)[1]

        blob_client = blob_service_client.get_blob_client(container=azure_blob_drop_storage_container, blob=blob_name)
        blob_properties = blob_client.get_blob_properties()
        tags = blob_properties.metadata.get("tags")
        if tags is not None:
            if isinstance(tags, str):
                tags_list = [tags]
            else:
                tags_list = tags.split(",")
        else:
            tags_list = []
        # Write the tags to cosmos db
        statusLog.update_document_tags(blob_path, tags_list)

        video_hash = generate_video_hash(blob_client)

        azure_config = {
            'AccountName': viAccountName,
            'ResourceGroup': azResourceGroup,
            'SubscriptionId': azSubscriptionId
        }

        client = init_video_indexer_client(azure_config)

        if check_video_exists(video_hash):
            print("Video already exists. Skipping...")
            video_id = get_existing_video_id(video_hash)
            logging.info("%s - Media upload skipped for %s: %s",
                            FUNCTION_NAME,
                            blob_path,
                            "Video: {file_name}, already exists.")
        else:
            supported_extensions: list = ['.mp4', '.mov', '.avi', '.mpg', '.wmv', '.wav', '.mp3']

            if file_extension not in supported_extensions:
                print(f"Unsupported video format: {file_name}. Skipping...")
                logging.error("%s - Media analysis failed for %s: %s",
                            FUNCTION_NAME,
                            blob_path,
                            "Unsupported video format: {file_name}.")
                statusLog.upsert_document(
                    blob_path,
                    f"{FUNCTION_NAME} - Media analysis failed: Unsupported video format: {file_name}.",
                    StatusClassification.ERROR,
                    State.ERROR,
                )
            else:
                print(f"Processing video: {file_name}")
                video_id = client.upload_url_async(
                    video_name=file_name,
                    video_url=blob_path_plus_sas,
                    wait_for_index=True
                )
                print(f"Video uploaded: {video_id} to AVI.")
                # Track the video hash
                upload_video_hash(video_id, blob_name, video_hash)

        ### Getting indexed videos prompt content ###
        video_prompt_content = client.get_prompt_content(video_id)

        ### Prepare language models ###
        language_models: LanguageModels = OpenAI()
        embeddings_size = language_models.get_embeddings_size()

        ### Adding prompt content sections ###
        account_details = client.get_account_details()
        sections_generator = get_sections_generator(video_id, video_prompt_content, account_details, embedding_cb=language_models.get_text_embeddings, embeddings_col_name=VECTOR_FIELD_NAME)
        verbose = True
        ### Creating new DB ###
        prompt_content_db : PromptContentDB = AzureVectorSearch()
        db_name = "vi-prompt-content-index"

        try:
            prompt_content_db.create_db(db_name, vector_search_dimensions=embeddings_size)
            #Section might already exist in index. Remove it first
            prompt_content_db.remove_section(db_name, file_name)

            prompt_content_db.add_sections_to_db(sections_generator, upload_batch_size=100, verbose=verbose)
            statusLog.upsert_document(
                blob_path,
                f"{FUNCTION_NAME} - Media added to index -> Media enrichment is complete.",
                StatusClassification.INFO,
                State.COMPLETE,
            )
            print("Done adding sections to DB. Exiting...")
        except Exception as err:
            statusLog.upsert_document(
                blob_path,
                f"{FUNCTION_NAME} - An error occurred while indexing - {str(err)}",
                StatusClassification.ERROR,
                State.ERROR,
            )

    except Exception as error:
        statusLog.upsert_document(
            blob_path,
            f"{FUNCTION_NAME} - An error occurred - {str(error)}",
            StatusClassification.ERROR,
            State.ERROR,
        )
    statusLog.save_document(blob_path) 
