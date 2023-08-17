from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession

from constants import GLOBAL_TIMEOUT

logger = logging.getLogger(__name__)
session: ClientSession | None = None  # stores aiohttp ClientSession object


@dataclass
class Response:
    """A class to store the response from an API request."""

    status: int
    headers: dict[str, str]
    data: Any


async def post_request(
    url: str,
    data: Any,
    headers: dict[str, str],
    timeout: int = GLOBAL_TIMEOUT,
    is_data_json: bool = True,
    chunked: bool = False,
) -> Response:
    """Make an asynchronous post request to an external API.

    Parameters
    ----------
    url :  str
        external API url
    data : Any
        request body
    headers : Dict[str, str]
        request headers
    timeout : int
        request timeout in seconds
    is_data_json : bool
        determines whether request body should be converted to json
    chunked : bool
        whether input data is chunked

    Returns
    ----------
    Response
        an object containing the response body, headers, and status
    """
    try:
        data_arg = "json" if is_data_json else "data"
        request_args = {
            data_arg: data,
            "headers": headers,
            "timeout": timeout,
        }

        if chunked:
            request_args["chunked"] = True

        async with session.post(url, **request_args) as resp:  # type: ignore

            if "application/json" in resp.headers.get("Content-Type", {}):
                response_data = await resp.json()
            else:
                response_data = await resp.read()

            if resp.status != 200:
                logger.error("Error with API request: %s", response_data)
        response = Response(resp.status, resp.headers, data=response_data)
    except Exception as error:  # pylint: disable=broad-except
        logger.error("API request failed with error: %s", repr(error))
        if "response" not in locals():
            response = Response(
                -1,
                {},
                {"error": dict.fromkeys(["type", "message"], type(error).__name__)},
            )
    return response
