import logging
from typing import Optional

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

logger = logging.getLogger(__name__)
logging.getLogger('WDM').setLevel(logging.WARNING)


def load_webdriver(headless: Optional[bool] = False) -> WebDriver:
    """
    A more robust way to load chrome webdriver (compatible with headless mode)
    Parameters
    ----------
    headless: whether to use headless mode
    Returns
    -------
    WebDriver
    """
    try:
        options = Options()
        if headless:
            options.headless = True
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
            )
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    except WebDriverException:

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        )

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    return driver
