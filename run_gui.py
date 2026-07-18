"""
Launch the Bosman AI Coach GUI.

    python run_gui.py

This is just a thin launcher — all the GUI code lives in gui/main_window.py,
and all the tactical reasoning lives in core/.
"""
from gui.main_window import main

if __name__ == "__main__":
    main()
