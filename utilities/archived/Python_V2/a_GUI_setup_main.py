import os, sys
import subprocess
import argparse

from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QLineEdit, QPushButton, QMessageBox, QProgressBar

from utilities.sqlcontroller import SqlController
from utilities.a_script_preprocess_setup import preprocess_setup
from utilities.file_location import FileLocationManager


class GUISetupMain(QWidget):
    def __init__(self, stack, parent=None):
        super(GUISetupMain, self).__init__(parent)

        # Stack specific info, determined from dropdown menu selection
        self.stack = stack
        self.fileLocationManager = FileLocationManager(self.stack)
        self.sqlController = SqlController()
        self.sqlController.get_animal_info(self.stack)
        self.stain = self.sqlController.histology.counterstain
        self.curr_step = self.sqlController.get_current_step_from_progress_ini(self.stack)

        # Init UI
        self.init_ui()

        # Set buttons functionality
        self.b_1_4.clicked.connect(lambda: self.click_button(self.b_1_4))
        self.b_1_6.clicked.connect(lambda: self.click_button(self.b_1_6))
        self.b_exit.clicked.connect(lambda: self.click_button(self.b_exit))

        # Update buttons
        self.update_buttons()

        # Center the GUI
        self.center()

    def init_ui(self):
        self.grid_buttons = QGridLayout()
        self.grid_bottom = QGridLayout()

        # Grid buttons
        self.b_1_4 = QPushButton("Setup Sorted Filenames")
        self.grid_buttons.addWidget(self.b_1_4)
        self.b_1_6 = QPushButton("Run automatic setup scripts")
        self.grid_buttons.addWidget(self.b_1_6)

        # Grid bottom
        self.progress = QProgressBar(self)
        self.progress.hide()
        self.grid_bottom.addWidget(self.progress)

        self.b_exit = QPushButton("Exit")
        self.b_exit.setDefault(True)
        self.grid_bottom.addWidget(self.b_exit)

        # Super grid
        self.super_grid = QGridLayout()
        self.super_grid.addLayout(self.grid_buttons, 1, 0)
        self.super_grid.addLayout(self.grid_bottom, 2, 0)
        self.setLayout(self.super_grid)
        self.setWindowTitle("Align to Active Brainstem Atlas - Setup Page")
        self.resize(1000, 450)

    def update_buttons(self):
        """
        Locates where you are in the pipeline by reading the brains_info/STACK_progress.ini

        Buttons corresponding to previous steps are marked as "completed", buttons corresponding
        to future steps are marked as "unpressable" and are grayed out.
        """

        self.stain = self.sqlController.histology.counterstain

        try:
            self.curr_step = self.sqlController.get_current_step_from_progress_ini(self.stack)
            print('format grid buttons current step is', self.curr_step)

            curr_step_index = ['1-4', '1-5', '1-6'].index(self.curr_step[:3])
            for index, button in enumerate([self.b_1_4, self.b_1_6]):
                if index <= curr_step_index + 1:
                    button.setEnabled(True)
                else:
                    button.setEnabled(False)

        # If there are no stacks/brains that have been started
        except KeyError:
            for button in [self.b_1_4, self.b_1_6]:
                button.setEnabled(False)

    def click_button(self, button):
        """
        If any of the "grid" buttons are pressed, this is the callback function.
        In this case, "grid" buttons have a one-to_one correspondance to the steps in the pipeline.
        The completion of each step means you move onto the next one.
        """
        # Setup/Create sorted filenames
        if button == self.b_1_4:
            try:
                subprocess.call(['python', 'a_GUI_setup_sorted_filenames.py', self.stack])
                self.update_buttons()
            except Exception as e:
                sys.stderr.write(str(e))
        # Run automatic scripts
        elif button == self.b_1_6:
            message = "This operation will take a long time."
            message += " Several minutes per image."
            QMessageBox.about(self, "Popup Message", message)
            preprocess_setup(self.stack, self.stain)

            #subprocess.call(['python', 'utilities/a_script_preprocess_1.py', self.stack, self.stain])
            subprocess.call(['python', 'a_script_preprocess_2.py', self.stack, self.stain])

            self.sqlController.set_step_completed_in_progress_ini(self.stack, '1-6_setup_scripts')

            """
            pipeline_status = get_pipeline_status(self.stack)
            if not 'preprocess_1' in pipeline_status and \
                    not 'preprocess_2' in pipeline_status and not 'setup' in pipeline_status:
                self.sqlController.set_step_completed_in_progress_ini(self.stack, '1-6_setup_scripts')
                sys.exit(app.exec_())

            pipeline_status = get_pipeline_status(self.stack)
            if pipeline_status == 'a_script_preprocess_3':
                self.sqlController.set_step_completed_in_progress_ini(self.stack, '1-6_setup_scripts')
                sys.exit(app.exec_())
            # else:
            #    print '\n\n\n\n'
            #    print 'pipeline_status:'
            ##    print pipeline_status
            #    print '\n\n\n\n'
            #    #set_step_completed_in_progress_ini( self.stack, '1-6_setup_scripts')
            print('finished in button 4')
            """
        elif button == self.b_exit:
            self.closeEvent(None)

        self.update_buttons()

    def center(self):
        """
        This function simply aligns the GUI to the center of your monitor.
        """
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def closeEvent(self, event):
        sys.exit(app.exec_())
        # close_main_gui( ex, reopen=True )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Setup main page')
    parser.add_argument("stack", type=str, help="stack name")
    args = parser.parse_args()
    stack = args.stack

    app = QApplication(sys.argv)
    ex = GUISetupMain(stack)
    ex.show()
    sys.exit(app.exec_())
