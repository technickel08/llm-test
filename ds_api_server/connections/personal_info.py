import requests
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)

def personal_info_wrapper(user_id):
    logger.info("Fetching personal information user_id : {}".format(user_id))
    try:
        url = "https://genwise-ai.agex.club/internal/v1/ai/user/profile?userId=149"

        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)

        return response.json()
    except Exception as e:
        logger.error("some exception occured - {}".format(str(e)))
        return None
