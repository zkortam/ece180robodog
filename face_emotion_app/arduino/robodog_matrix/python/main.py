"""Keep the Arduino App runtime alive; voice control runs in systemd."""
import time

from arduino.app_utils import App


def loop():
    time.sleep(60)


App.run(user_loop=loop)
