import asyncio
import logging
import sys

import uvicorn

from pc_liveagent.config import PORT

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
    uvicorn.run(
        "pc_liveagent.main:app",
        host="127.0.0.1",
        port=PORT,
        reload=sys.platform != "win32",
        log_config=None,
    )
