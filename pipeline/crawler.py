import os
import time
import json
import random
import logging
from tqdm.auto import tqdm
from typing import Optional
from seqlbtoolkit.text import substring_mapping
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from chempp.constants import CHAR_TO_HTML_LBS
from .utils import load_webdriver

logger = logging.getLogger(__name__)


def scroll_action(driver, height: Optional[int] = 720):
    """
    Scroll the webpage by height

    Parameters
    ----------
    driver: selenium driver
    height: height to scroll

    Returns
    -------
    None
    """
    driver.execute_script(f"window.scrollTo(0, {height})")
    return None


def scroll_to_bottom(driver):
    """
    Scroll the webpage by height

    Parameters
    ----------
    driver: selenium driver

    Returns
    -------
    None
    """
    driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight);")
    return None


def scroll_page(driver, interval: Optional[float] = 0.1, height: Optional[int] = 1080):
    """
    Scroll the webpage so that it is fully loaded

    Parameters
    ----------
    driver: selenium driver
    interval: interval in second between each scroll action
    height: height to scroll in each scroll action

    Returns
    -------
    None
    """
    for t in range(round(random.uniform(2, 5))):
        scroll_action(driver, height * (t + 1))
        time.sleep(interval)

    time.sleep(3 * interval)
    scroll_to_bottom(driver)
    return None


def download_pages(dois: list,
                   save_dir: str,
                   headless: Optional[bool] = False,
                   driver_timeout: Optional[float] = 20):
    """
    Download crawl_results given dois.
    Notice that this function only download HTML plain text while ignoring figures, icons or css files.

    Parameters
    ----------
    dois: a list of doi
    save_dir: folder to save the result
    headless: whether download pages in headless mode
    driver_timeout: maximum time for a webpage to load
    Returns
    -------
    None
    """
    # Make sure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    failed_dois = list()

    for doi in tqdm(dois):
        driver = load_webdriver(headless=headless)
        driver.implicitly_wait(driver_timeout)

        try:
            driver.get(f"https://doi.org/{doi}")

            time.sleep(0.3)
            scroll_page(driver)

            if doi.startswith('10.1039'):
                try:
                    driver.find_element(By.LINK_TEXT, 'Article HTML').click()

                    time.sleep(0.3)
                    scroll_page(driver)

                except NoSuchElementException:
                    pass

            # Parse page elements to get webpage links
            html_content = driver.page_source
            file_name = substring_mapping(doi, CHAR_TO_HTML_LBS) + '.html'
            with open(os.path.join(save_dir, file_name), 'w', encoding='utf-8') as f:
                f.write(html_content)

        except TimeoutException:
            failed_dois.append(doi)
            logger.warning(f"Failed to retrieve doi {doi} due to time-out.")

        except Exception as e:
            failed_dois.append(doi)
            logger.exception(f"Failed to retrieve webpage {doi}, error: {e}")

        finally:
            driver.quit()

    return failed_dois


def crawl(args):

    if args.doi_file_path:
        with open(args.doi_file_path, 'r', encoding='utf-8') as f:
            dois = json.load(f)

        dois = [d[16:] for d in dois if 'http' in d]

    failed = download_pages(dois=dois, save_dir=args.raw_article_dir, headless=True)
    if failed:
        failed_str = '\n'.join(failed)
        logger.warning(f"Failed to retrieve the following articles:\n{failed_str}.")

