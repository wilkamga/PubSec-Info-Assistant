import json
import logging
import os
import time
from typing import Optional

import azure.functions as func
import requests
from azure.storage.blob import BlobServiceClient
from shared_code.status_log import State, StatusClassification, StatusLog
from shared_code.utilities import Utilities, MediaType

from vi_search.language_models.azure_openai import OpenAI
from vi_search.prompt_content_db.azure_search import AzureVectorSearch
from vi_search.prep_scenes import get_sections_generator
from vi_search.language_models.language_models import LanguageModels
from vi_search.prompt_content_db.prompt_content_db import PromptContentDB, VECTOR_FIELD_NAME
from vi_search.vi_client.video_indexer_client import init_video_indexer_client, VideoIndexerClient

from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField

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


def wait_for_video_processing(client: VideoIndexerClient, video_id: dict, get_insights: bool = False,
                              timeout: int = 600) -> Optional[dict[str, dict]]:
    start = time.time()

    insights = {}
    while True:
        res = client.is_video_processed(video_id)
        if res:
            print(f"Video {video_id} processing completed.")
            if get_insights:
                insights = client.get_video_async(video_id)
            break

        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Timeout reached.")

        if elapsed % 20 == 0:
            print(
                f"Elapsed time: {time.time() - start} seconds. Waiting for video to process.")

        time.sleep(1)

    print(f"Video processing completed, took {time.time() - start} seconds")

    if get_insights:
        return insights


def detect_language(text):
    data = {
        "kind": "LanguageDetection",
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "text": text[:MAX_CHARS_FOR_DETECTION]
                }
            ]
        }
    }

    response = requests.post(
        API_DETECT_ENDPOINT, headers=translator_api_headers, json=data
    )
    if response.status_code == 200:
        print(response.json())
        detected_language = response.json(
        )["results"]["documents"][0]["detectedLanguage"]["iso6391Name"]
        detection_confidence = response.json(
        )["results"]["documents"][0]["detectedLanguage"]["confidenceScore"]

    return detected_language, detection_confidence


def translate_text(text, target_language):
    data = [{"text": text}]
    params = {"to": target_language}

    response = requests.post(
        API_TRANSLATE_ENDPOINT, headers=translator_api_headers, json=data, params=params
    )
    if response.status_code == 200:
        translated_content = response.json()[0]["translations"][0]["text"]
        return translated_content
    else:
        raise Exception(response.json())


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

        # Run the image through the Computer Vision service
        file_name, file_extension, file_directory = utilities.get_filename_and_extension(
            blob_path)
        blob_path_plus_sas = utilities.get_blob_and_sas(blob_path)

        data = {"url": f"{blob_path_plus_sas}"}

        azure_config = {
            'AccountName': viAccountName,
            'ResourceGroup': azResourceGroup,
            'SubscriptionId': azSubscriptionId
        }

        client = init_video_indexer_client(azure_config)

        supported_extensions: list = ['.mp4', '.mov', '.avi']

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
        """ if client.get_video_async(video_id):
            print(f"Media already exist: {file_name}. Skipping...") """
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
        prompt_content_db.create_db(db_name, vector_search_dimensions=embeddings_size)
        prompt_content_db.add_sections_to_db(sections_generator, upload_batch_size=100, verbose=verbose)

        print("Done adding sections to DB. Exiting...")
        
        # Upload the output as a chunk to match document model
        """ utilities.write_chunk(
            myblob_name=blob_path,
            myblob_uri=blob_uri,
            file_number=0,
            chunk_size=utilities.token_count(text_image_summary),
            chunk_text=text_image_summary,
            page_list=[0],
            section_name="",
            title_name=file_name,
            subtitle_name="",
            file_class=MediaType.IMAGE
        ) """

        statusLog.upsert_document(
            blob_path,
            f"{FUNCTION_NAME} - Media enrichment is complete",
            StatusClassification.DEBUG,
            State.QUEUED,
        )

    except Exception as error:
        statusLog.upsert_document(
            blob_path,
            f"{FUNCTION_NAME} - An error occurred - {str(error)}",
            StatusClassification.ERROR,
            State.ERROR,
        )

    """ try:
        file_name, file_extension, file_directory = utilities.get_filename_and_extension(
            blob_path)

        # Get the tags from metadata on the blob
        path = file_directory + file_name + file_extension
        blob_service_client = BlobServiceClient.from_connection_string(
            azure_blob_connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=azure_blob_drop_storage_container, blob=path)
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

        # Only one chunk per image currently.
        chunk_file = utilities.build_chunk_filepath(
            file_directory, file_name, file_extension, '0')

        index_section(index_content, file_name, file_directory[:-1], statusLog.encode_document_id(
            chunk_file), chunk_file, blob_path, blob_uri, tags_list)

        statusLog.upsert_document(
            blob_path,
            f"{FUNCTION_NAME} - Media added to index.",
            StatusClassification.INFO,
            State.COMPLETE,
        )
    except Exception as err:
        statusLog.upsert_document(
            blob_path,
            f"{FUNCTION_NAME} - An error occurred while indexing - {str(err)}",
            StatusClassification.ERROR,
            State.ERROR,
        )

    statusLog.save_document(blob_path) """


def index_section(index_content, file_name, file_directory, chunk_id, chunk_file, blob_path, blob_uri, tags):
    """ Pushes a batch of content to the search index
    """

    index_chunk = {}
    batch = []
    index_chunk['id'] = chunk_id
    azure_datetime = datetime.now().astimezone().isoformat()
    index_chunk['processed_datetime'] = azure_datetime
    index_chunk['file_name'] = blob_path
    index_chunk['file_uri'] = blob_uri
    index_chunk['folder'] = file_directory
    index_chunk['title'] = file_name
    index_chunk['content'] = index_content
    index_chunk['pages'] = [0]
    index_chunk['chunk_file'] = chunk_file
    index_chunk['file_class'] = MediaType.MEDIA
    index_chunk['tags'] = tags
    batch.append(index_chunk)

    search_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
                                 index_name=AZURE_SEARCH_INDEX,
                                 credential=SEARCH_CREDS)

    search_client.upload_documents(documents=batch)
