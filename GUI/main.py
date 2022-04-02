import sys
from PyQt5.QtWidgets import QApplication
from SplashScreen import SplashScreen

if __name__ == '__main__':
    app = QApplication([])
    window = SplashScreen()
    window.show()
    sys.exit(app.exec_())
