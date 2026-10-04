"""Qt stylesheet for the ECG viewer."""

APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #f8fafc;
    color: #0f172a;
    font-size: 13px;
}
QLabel#Title {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}
QLabel#FileLabel {
    color: #475569;
    padding-left: 8px;
}
QLabel#SectionTitle {
    font-weight: 700;
    color: #0f172a;
}
QLabel#Subtle {
    color: #475569;
}
QLabel#Stats {
    background: #ffffff;
    border: 1px solid #dbe3ea;
    border-radius: 6px;
    padding: 8px 10px;
    color: #1e293b;
}
QFrame#Panel {
    background: #ffffff;
    border: 1px solid #dbe3ea;
    border-radius: 6px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 6px 10px;
}
QPushButton:hover {
    background: #f1f5f9;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background: #e2e8f0;
}
QPushButton:disabled {
    color: #94a3b8;
    background: #f8fafc;
}
QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 4px 6px;
    min-width: 100px;
}
QLineEdit {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 5px 7px;
}
QCheckBox {
    spacing: 6px;
}
"""
