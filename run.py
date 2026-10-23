import asyncio
import logging
import sys

import uvicorn

from pc_liveagent.config import APPLICATION_AFTER_SIGNIN, APPLICATION_TEST_MODE, PORT

# Playwright needs a Proactor loop on Windows. reload=True uses a Selector
# loop and then Chrome never starts (NotImplementedError / long hang).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logging.getLogger("pc_liveagent").info(
        "runtime config POINTCARE_APPLICATION_AFTER_SIGNIN=%s "
        "POINTCARE_APPLICATION_TEST_MODE=%s",
        APPLICATION_AFTER_SIGNIN,
        APPLICATION_TEST_MODE,
    )
    uvicorn.run(
        "pc_liveagent.main:app",
        host="127.0.0.1",
        port=PORT,
        reload=sys.platform != "win32",
        log_config=None,
    )
