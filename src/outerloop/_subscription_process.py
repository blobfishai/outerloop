"""Private native-process supervisor; no provider or research policy lives here.

The controller owns a liveness pipe. EOF (including controller SIGKILL) kills
the native process group. Native stdin/stdout remain direct controller pipes.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys


def main() -> int:
    watch_fd = int(sys.argv[1])
    done_fd = int(sys.argv[2])
    lock_fds = tuple(int(value) for value in sys.argv[3].split(","))
    command = sys.argv[4:]
    # The controller can die during launch. Never spawn after its pipe closed.
    if select.select([watch_fd], [], [], 0)[0]:
        return 125
    # Stay in the supervisor's group. The controller knows this group from its
    # own Popen receipt, never from untrusted native stdout. Host death therefore
    # cannot detach an old writer into a different group.
    process = subprocess.Popen(command, pass_fds=lock_fds)
    try:
        reported = False
        while True:
            if select.select([watch_fd], [], [], 0.05)[0]:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            if not reported and process.poll() is not None:
                # A separate inherited descriptor carries the trusted exit code;
                # native stdout cannot forge it. Keep watching after leader exit
                # until the controller cleans up or dies. The group still owns
                # this PID throughout the handoff, even with background children.
                os.write(done_fd, f"{process.returncode}\n".encode())
                os.close(done_fd)
                reported = True
    except BaseException:
        os.killpg(os.getpgrp(), signal.SIGKILL)
        raise


if __name__ == "__main__":
    sys.exit(main())
