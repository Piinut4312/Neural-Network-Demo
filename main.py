from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QAction, QPainter, QKeySequence, QBrush, QColor, QPen, QFont
from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QMainWindow, QApplication, QSizePolicy, QLabel, QLineEdit, QCheckBox, QPushButton, QComboBox, QProgressBar, QTreeWidget, QTreeWidgetItem, QTabWidget, QDialog, QDialogButtonBox, QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout
from PySide6.QtCharts import QChart, QChartView, QScatterSeries, QLineSeries, QValueAxis
from dataset_loader import DatasetLoader
from perceptron import MultiLayerPerceptron
from function import FUNCTIONS
from metrics import ACCURACY
from learning_rate_scheduler import LR_SCHEDULERS
import os
from math import log10
import numpy as np

class ErrorDialog(QDialog):

    def __init__(self, parent=None, title='', message=''):
        super().__init__()
        self.setWindowTitle(title)

        self.layout = QVBoxLayout()
        message = QLabel(message)
        self.button = QPushButton("Ok")
        self.button.clicked.connect(self.accept)
        self.layout.addWidget(message)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)


class ModelVisualizer(QWidget):

    def __init__(self):
        super(ModelVisualizer, self).__init__()
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.neurons = []
        self.model = None


    class _Link:

        def __init__(self, x1, y1, x2, y2, label=""):
            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2
            self.label = label

        def get_label_pos(self, c=0.5):
            return c*self.x1+(1-c)*self.x2, c*self.y1+(1-c)*self.y2
        

    class _Neuron:

        def __init__(self, x, y, r, bias=False, highlight=False):
            self.x = x
            self.y = y
            self.r = r
            self.bias = bias
            self.highlight = highlight
            self.links = []
            self.model = None
        
        def add_link(self, x, y, label=""):
            self.links.append(ModelVisualizer._Link(x, y, self.x, self.y+self.r/2, label))


    def build_network_graph(self):
        # Calculate the layout of neurons and links to display in the model visualizer
        geometry = self.geometry()

        r = 40
        model_w = geometry.width()*1
        model_h = geometry.height()*0.8
        model_left = (geometry.width()-model_w)*0.5
        model_top = (geometry.height()-model_h)*0.5

        self.neurons = []
        if self.model is not None:
            hgap = model_w/len(self.model.dims)
            last_layer_pos = []
            for i, n in enumerate(self.model.dims):
                if i < len(self.model.dims)-1:
                    m = n+1
                else:
                    m = n
                vgap = model_h/m
                for j in range(m):
                    neuron_x = model_left+hgap/2+hgap*i
                    neuron_y = model_top+vgap/2+j*vgap
                    neuron = ModelVisualizer._Neuron(neuron_x, neuron_y, 40)
                    last_layer_pos.append((neuron_x, neuron_y))
                    
                    if j > 0 or i == len(self.model.dims)-1:
                        if i > 0:
                            for k in range(self.model.dims[i-1]+1):
                                endpoint_x = last_layer_pos[k][0]+r
                                endpoint_y = last_layer_pos[k][1]+r/2
                                neuron.add_link(endpoint_x, endpoint_y, "{weight:.3f}".format(weight=self.model.W[i-1][j-1][k]))
                    else:
                        neuron.bias = True
         
                    self.neurons.append(neuron)
                    
                if i > 0:
                    last_layer_pos = last_layer_pos[self.model.dims[i-1]+1:]
        self.update()
    

    def paintEvent(self, e):
        painter = QPainter(self)
        rect = QRect(0, 0, painter.device().width(), painter.device().height())
        background = QBrush()
        background.setColor(Qt.white)
        background.setStyle(Qt.SolidPattern)
        painter.fillRect(rect, background)

        solid_pen = QPen(Qt.lightGray, 3, Qt.SolidLine)
        dash_pen = QPen(Qt.lightGray, 3, Qt.DashLine)
        hint_pen = QPen(Qt.black, 3, Qt.SolidLine)
        highlight_link_pen = QPen(Qt.darkGray, 4, Qt.SolidLine)
        highlight_label_pen = QPen(Qt.blue, 4, Qt.SolidLine)
        font = painter.font()
        painter.setPen(hint_pen)
        font.setPointSize(14)

        painter.setFont(font)
        if len(self.neurons) == 0:
            painter.drawText(
                0, 
                0,
                self.geometry().width(),
                self.geometry().height(),
                Qt.AlignmentFlag.AlignCenter,
                "You haven't trained the model yet.\nTrain the model first to update the model visualization."
            )
            return

        painter.drawText(
            0, 
            0,
            self.geometry().width(),
            self.geometry().height(),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            "Click on the neurons to see the value of weights."
        )

        painter.setPen(solid_pen)
        hightlighted_neuron = None
        for neuron in self.neurons:
            painter.setPen(solid_pen)
            if neuron.bias:
                font.setPointSize(18)
                painter.setFont(font)
                painter.drawText(
                    neuron.x, 
                    neuron.y,
                    neuron.r,
                    neuron.r,
                    Qt.AlignmentFlag.AlignCenter,
                    "-1"
                )
                painter.setPen(dash_pen)
            if neuron.highlight:
                hightlighted_neuron = neuron
                painter.setPen(highlight_link_pen)

            painter.drawEllipse(neuron.x, neuron.y, neuron.r, neuron.r)

            for link in neuron.links:
                if not neuron.highlight:
                    painter.setPen(solid_pen)
                    painter.drawLine(link.x1, link.y1, link.x2, link.y2)

        if hightlighted_neuron is not None:
            for link in hightlighted_neuron.links:
                painter.setPen(highlight_link_pen)
                painter.drawLine(link.x1, link.y1, link.x2, link.y2)

            for link in hightlighted_neuron.links:
                text_x, text_y = link.get_label_pos(c=0.5)
                font.setPointSize(14)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(highlight_label_pen)
                painter.drawText(
                    text_x, 
                    text_y,
                    link.label
                )
                font.setBold(False)

        painter.end()


    def mousePressEvent(self, e):
        pos = e.position()
        for neuron in self.neurons:
            if (neuron.x-pos.x()+neuron.r/2)**2+(neuron.y-pos.y()+neuron.r/2)**2 <= (neuron.r/2)**2:
                neuron.highlight = True
            else:
                neuron.highlight = False
        self.update()

                
class MainWidget(QWidget):

    statusMessage = Signal(str)

    def __init__(self):
        QWidget.__init__(self)

        self.data_dict = {"basic":[s.removesuffix(".txt") for s in os.listdir("datasets/basic/")], "extra": [s.removesuffix(".txt") for s in os.listdir("datasets/extra/")]}
        self.data_list = list(self.data_dict.values())
        self.data_list = self.data_list[0] + self.data_list[1]
        self.current_data_loader = None
        self.data_loaders = {k: None for k in self.data_list}

        self.model = None

        # Data selection
        data_setting_groupbox = QGroupBox("Data settings")

        self.example_tree = QTreeWidget()
        self.example_tree.setColumnCount(1)
        self.example_tree.setHeaderLabels(["Name"])
        tree_items = []
        for k, v in self.data_dict.items():
            item = QTreeWidgetItem([k])
            for f in v:
                child = QTreeWidgetItem([f, k])
                item.addChild(child)
            tree_items.append(item)
        self.example_tree.insertTopLevelItems(0, tree_items)
        self.example_tree.currentItemChanged.connect(self.load_data)

        self.num_data_text = QLabel("Number of data: 0")
        self.data_dim_text = QLabel("Dimension of data: 0")
        self.num_label_text = QLabel("Number of labels: 0")
        self.split_ratio_input = QLineEdit("0.33")
        self.standardize_checkbox = QCheckBox()
        self.standardize_checkbox.stateChanged.connect(self.redraw_charts)

        # Model and training settings
        model_setting_group = QGroupBox("Model and training settings")
        self.model_shape_input = QLineEdit("2,1")
        self.activation_func_combo = QComboBox()
        self.activation_func_combo.addItems(["Sigmoid", "ReLU", "Leaky ReLU", "Tanh"])
        self.epoch_input = QLineEdit("10")
        self.learning_rate_input = QLineEdit("0.5")
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems(["Constant", "Reciprocal"])
        self.momentum_input = QLineEdit("0.5")
        self.early_stop_input = QLineEdit("0.9")
        self.train_button = QPushButton("Train model")
        self.train_button.clicked.connect(self.train_model)
        self.train_progress_bar = QProgressBar()

        # Model testing
        model_testing_group = QGroupBox("Test")
        self.test_data_input = QLineEdit("0.0, 0.0")
        self.test_data_input.textEdited.connect(self.clear_test_info)
        self.sample_button = QPushButton("Sample from loaded data")
        self.sample_button.clicked.connect(self.sample_data)
        self.compute_button = QPushButton("Compute")
        self.compute_button.clicked.connect(self.compute_test_data)
        self.test_output = QLabel("Model output: 0.0")
        self.test_output.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.test_prediction = QLabel()
        self.test_prediction.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.test_actual_label = QLabel()
        self.test_actual_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.clear_test_info()

        # Data charts
        self.data_chart = QChart()
        self.data_chart.setAnimationOptions(QChart.AllAnimations)

        self.data_chart_x_axis = QValueAxis()
        self.data_chart_x_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.data_chart_x_axis.setTickInterval(1)

        self.data_chart_y_axis = QValueAxis()
        self.data_chart_y_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.data_chart_y_axis.setTickInterval(1)

        self.data_chart.addAxis(self.data_chart_x_axis, Qt.AlignmentFlag.AlignBottom)
        self.data_chart.addAxis(self.data_chart_y_axis, Qt.AlignmentFlag.AlignLeft)

        self.data_chart_view = QChartView(self.data_chart)
        self.data_chart_view.setRenderHint(QPainter.Antialiasing)

        # Train result chart
        self.train_result_chart = QChart()
        self.train_result_chart.setAnimationOptions(QChart.AllAnimations)

        self.train_result_chart_x_axis = QValueAxis()
        self.train_result_chart_x_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.train_result_chart_x_axis.setTickInterval(1)

        self.train_result_chart_y_axis = QValueAxis()
        self.train_result_chart_y_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.train_result_chart_y_axis.setTickInterval(1)

        self.train_result_chart.addAxis(self.train_result_chart_x_axis, Qt.AlignmentFlag.AlignBottom)
        self.train_result_chart.addAxis(self.train_result_chart_y_axis, Qt.AlignmentFlag.AlignLeft)

        self.train_result_chart_view = QChartView(self.train_result_chart)
        self.train_result_chart_view.setRenderHint(QPainter.Antialiasing)

        # Learning curve chart
        self.learn_chart = QChart()
        self.learn_chart.setAnimationOptions(QChart.AllAnimations)

        self.learn_chart_x_axis = QValueAxis()
        self.learn_chart_x_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.learn_chart_x_axis.setTickInterval(1)
        self.learn_chart_x_axis.setTitleText("Epoch")

        self.learn_chart_y_axis = QValueAxis()
        self.learn_chart_y_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.learn_chart_y_axis.setTickInterval(0.2)
        self.learn_chart_y_axis.setRange(0, 1.1)

        self.learn_chart.addAxis(self.learn_chart_x_axis, Qt.AlignmentFlag.AlignBottom)
        self.learn_chart.addAxis(self.learn_chart_y_axis, Qt.AlignmentFlag.AlignLeft)

        self.learn_chart_view = QChartView(self.learn_chart)
        self.learn_chart_view.setRenderHint(QPainter.Antialiasing)

        self.model_visualizer = ModelVisualizer()

        chart_tabs = QTabWidget()
        chart_tabs.setTabPosition(QTabWidget.North)

        chart_tabs.addTab(self.data_chart_view, "Data visualization (2D)")
        chart_tabs.addTab(self.train_result_chart_view, "Train result (2D)")
        chart_tabs.addTab(self.learn_chart_view, "Learning curve")
        chart_tabs.addTab(self.model_visualizer, "Model visualization")

        chart_tabs.currentChanged.connect(self.model_visualizer.build_network_graph)

        # Layouts
        main_layout = QHBoxLayout()
        settings_layout = QHBoxLayout()
        data_setting_layout = QVBoxLayout()
        model_setting_form = QFormLayout()
        model_setting_layout = QVBoxLayout()
        test_layout = QVBoxLayout()

        main_layout.addLayout(settings_layout)
        main_layout.addWidget(chart_tabs)
        settings_layout.addWidget(data_setting_groupbox)
        central_layout = QVBoxLayout()
        settings_layout.addLayout(central_layout)
        central_layout.addWidget(model_setting_group)
        central_layout.addWidget(model_testing_group)

        data_setting_groupbox.setLayout(data_setting_layout)
        data_setting_layout.addWidget(QLabel("Dataset"))
        data_setting_layout.addWidget(self.example_tree)
        data_setting_layout.addWidget(self.num_data_text)
        data_setting_layout.addWidget(self.data_dim_text)
        data_setting_layout.addWidget(self.num_label_text)
        
        train_test_split_layout = QHBoxLayout()
        data_setting_layout.addLayout(train_test_split_layout)
        train_test_split_layout.addWidget(QLabel("Portion of data for testing:"))
        train_test_split_layout.addWidget(self.split_ratio_input)

        standardize_layout = QHBoxLayout()
        data_setting_layout.addLayout(standardize_layout)
        standardize_layout.addWidget(QLabel("Standardize input data:"))
        standardize_layout.addWidget(self.standardize_checkbox)

        model_setting_group.setLayout(model_setting_layout)
        model_setting_layout.addLayout(model_setting_form)
        model_setting_form.addRow(QLabel("Model shape"), self.model_shape_input)
        model_setting_form.addRow(QLabel("Activation function"), self.activation_func_combo)
        model_setting_form.addRow(QLabel("Epochs"), self.epoch_input)
        model_setting_form.addRow(QLabel("Learning rate"), self.learning_rate_input)
        model_setting_form.addRow(QLabel("Learning rate scheduler"), self.lr_scheduler_combo)
        model_setting_form.addRow(QLabel("Momentum"), self.momentum_input)
        model_setting_form.addRow(QLabel("Early stopping accuracy"), self.early_stop_input)
        model_setting_layout.addWidget(self.train_button)
        model_setting_layout.addWidget(self.train_progress_bar)

        model_testing_group.setLayout(test_layout)
        test_input_layout = QHBoxLayout()
        test_input_layout.addWidget(QLabel("Test data:"))
        test_input_layout.addWidget(self.test_data_input)
        test_layout.addLayout(test_input_layout)
        test_layout.addWidget(self.sample_button)
        test_layout.addWidget(self.compute_button)
        test_layout.addWidget(self.test_output)
        test_layout.addWidget(self.test_prediction)
        test_layout.addWidget(self.test_actual_label)

        self.setLayout(main_layout)


    def redraw_data_chart(self):
        if self.current_data_loader is None:
            return
        if self.current_data_loader.data.shape[0] != 2:
            self.data_chart.removeAllSeries()
            return

        standardize = self.standardize_checkbox.checkState() == Qt.CheckState.Checked
        if standardize:
            bound_min, bound_max = self.current_data_loader.standardized_bound_box
            self.data_chart_x_axis.setRange(float(bound_min[0])-1, float(bound_max[0])+1)
            self.data_chart_y_axis.setRange(float(bound_min[1])-1, float(bound_max[1])+1)
        else:
            bound_min, bound_max = self.current_data_loader.bound_box
            self.data_chart_x_axis.setRange(float(bound_min[0])-1, float(bound_max[0])+1)
            self.data_chart_y_axis.setRange(float(bound_min[1])-1, float(bound_max[1])+1)
        
        self.data_chart.removeAllSeries()

        self.data_series = {k: QScatterSeries() for k in self.current_data_loader.label_set}
        for i, series in self.data_series.items():
            self.data_chart.addSeries(series)
            series.attachAxis(self.data_chart_x_axis)
            series.attachAxis(self.data_chart_y_axis)
            series.setName("Label="+str(i))

        for data, label in self.current_data_loader.get_tuples(standardize):
            self.data_series[label].append(*data)


    def redraw_train_result_chart(self):
        if self.current_data_loader is None or self.model is None:
            return
        if self.current_data_loader.data.shape[0] != 2:
            self.train_result_chart.removeAllSeries()
            return

        standardize = self.standardize_checkbox.checkState() == Qt.CheckState.Checked
        if standardize:
            bound_min, bound_max = self.current_data_loader.standardized_bound_box
            self.train_result_chart_x_axis.setRange(float(bound_min[0])-1, float(bound_max[0])+1)
            self.train_result_chart_y_axis.setRange(float(bound_min[1])-1, float(bound_max[1])+1)
        else:
            bound_min, bound_max = self.current_data_loader.bound_box
            self.train_result_chart_x_axis.setRange(float(bound_min[0])-1, float(bound_max[0])+1)
            self.train_result_chart_y_axis.setRange(float(bound_min[1])-1, float(bound_max[1])+1)
        
        self.train_result_chart.removeAllSeries()

        self.train_result_series = {k: QScatterSeries() for k in self.current_data_loader.label_set}
        for i, series in self.train_result_series.items():
            self.train_result_chart.addSeries(series)
            series.attachAxis(self.train_result_chart_x_axis)
            series.attachAxis(self.train_result_chart_y_axis)
            series.setName("Label="+str(i))

        for data, _ in self.current_data_loader.get_tuples(standardize):
            label = self.model.predict(data, classes=len(self.current_data_loader.label_set))
            self.train_result_series[float(np.squeeze(label))].append(*data)

    
    def redraw_charts(self):
        self.redraw_data_chart()
        self.redraw_train_result_chart()


    def load_data(self, selected_item: QTreeWidgetItem, previous_item: QTreeWidgetItem):
        file_name = selected_item.data(0, 0)
        folder = selected_item.data(1, 0)
        if folder is None:
            return
        if self.data_loaders[file_name] is None:
            self.data_loaders[file_name] = DatasetLoader("datasets/"+folder+"/"+file_name+".txt", encoding='scale')
        self.statusMessage.emit('Successfully loaded data: "'+file_name+'"')
        self.current_data_loader = self.data_loaders[file_name]
        self.num_data_text.setText("Number of data: "+str(self.current_data_loader.num_cases))
        self.data_dim_text.setText("Dimesion of data: "+str(self.current_data_loader.data_dim))
        self.num_label_text.setText("Number of labels: "+str(len(self.current_data_loader.label_set)))
        self.redraw_charts()


    def show_error_dialog(self, title, message):
        app.beep()
        dialog = ErrorDialog(self, title, message)
        return dialog.exec()
        

    def parse_model_settings(self):
        model_shape = []
        for x in self.model_shape_input.text().replace(' ', '').split(','):
            try:
                model_shape.append(int(x))
            except ValueError:
                self.show_error_dialog('Model setting error', 'The format of model shape must be: "x1, x2,..., xn", where x1~xn are positive integers!')
                return
            if model_shape[-1] <= 0:
                self.show_error_dialog('Model setting error', "The number of neurons must be positive integers!")
                return

        if len(model_shape) < 2:
            self.show_error_dialog('Model setting error', "The depth of the model must be at least 2! (Including input layer and output layer)")
            return
        
        activation_func = self.activation_func_combo.currentText()

        try:
            epoch = int(self.epoch_input.text())
        except ValueError:
            self.show_error_dialog('Model setting error', "Epoch must be set to a positive integer!")
            return
        if epoch <= 0:
            self.show_error_dialog('Model setting error', "Epoch must be set to a positive integer!")
            return
        
        try:
            learning_rate = float(self.learning_rate_input.text())
        except ValueError:
            self.show_error_dialog('Model setting error', "Invalid learning rate setting!")
            return
        
        lr_scheduler = self.lr_scheduler_combo.currentText()

        try:
            momentum = float(self.momentum_input.text())
        except ValueError:
            self.show_error_dialog('Model setting error', "Invalid momentum setting!")
            return
        
        try:
            early_stop_acc = float(self.early_stop_input.text())
        except ValueError:
            self.show_error_dialog('Model setting error', "Invalid early stopping setting!")
            return
        
        try:
            split_ratio = float(self.split_ratio_input.text())
        except ValueError:
            self.show_error_dialog('Model setting error', "Invalid train test split ratio!")
            return
        if not 0 < split_ratio < 1:
            self.show_error_dialog('Model setting error', "Invalid train test split ratio!")
            return
        
        return model_shape, activation_func, epoch, learning_rate, lr_scheduler, momentum, early_stop_acc, split_ratio


    def clear_test_info(self):
        self.test_output.setText("Model output: Unknown")
        self.test_prediction.setText("Predicted label: Unknown")
        self.test_actual_label.setText("Actual label: Unknown")


    def parse_test_data(self):
        data = []
        for x in self.test_data_input.text().replace(' ', '').split(','):
            try:
                data.append(float(x))
            except ValueError:
                self.show_error_dialog('Testing error', 'Invalid test data!')
                return
            
        if len(data) <= 0:
            self.show_error_dialog('Testing error', "Invalid test data!")
            return
        
        data = np.expand_dims(np.array(data), axis=1)
        return data
    

    def sample_data(self):
        if self.current_data_loader is None:
            self.show_error_dialog('Testing error', "Please select a dataset first!")
            return
        
        self.clear_test_info()

        standardize = self.standardize_checkbox.checkState() == Qt.CheckState.Checked
        data, label = self.current_data_loader.sample(standardize)
        data_str = ""
        for i in range(data.shape[0]):
            data_str += "{:.3f}".format(data[i])
            if i < data.shape[0]-1:
                data_str += ", "

        self.test_data_input.setText(data_str)
        self.test_actual_label.setText("Acutal label: "+str(label[0]))
    

    def compute_test_data(self):
        data = self.parse_test_data()
        if data is None:
            return
        
        if self.model is None:
            self.show_error_dialog('Testing error', "Please train a model first!")
            return
        
        self.test_output.setText("Model output: "+"{:.3f}".format(np.squeeze(self.model.forward(data)))+"...")
        self.test_prediction.setText("Predicted label: "+str(np.squeeze(self.model.predict(data, len(self.current_data_loader.label_set)))))

    def train_model(self):
        
        model_setting = self.parse_model_settings()
        if model_setting is None:
            return
        
        if self.current_data_loader is None:
            self.show_error_dialog("Model setting error", "Please select a dataset first!")
            return
        
        model_shape, activation_func, epoch, learning_rate, lr_scheduler, momentum, early_stop_acc, split_ratio = model_setting

        if model_shape[0] != self.current_data_loader.data_dim:
            self.show_error_dialog("Model setting error", "Input size does not match with the dimensions of data!")
            return

        standardize = self.standardize_checkbox.checkState() == Qt.CheckState.Checked
        self.model = MultiLayerPerceptron(model_shape, FUNCTIONS[activation_func])
        
        self.model_visualizer.model = self.model
        self.model_visualizer.build_network_graph()

        x_train, y_train, x_test, y_test = self.current_data_loader.train_test_split(split_ratio, standardize)

        train_accuracy_curve = QLineSeries()
        train_accuracy_curve.setName("Train accuracy")
        test_accuracy_curve = QLineSeries()
        test_accuracy_curve.setName("Test accuracy")
        train_error_curve = QLineSeries()
        train_error_curve.setName("Train MSE")
        test_error_curve = QLineSeries()
        test_error_curve.setName("Test MSE")

        num_classes = len(self.current_data_loader.label_set)

        train_accuracy_curve.append(0, self.model.evaluate(x_train, y_train, num_classes, metrics=[ACCURACY])['accuracy'])
        test_accuracy_curve.append(0, self.model.evaluate(x_test, y_test, num_classes, metrics=[ACCURACY])['accuracy'])
        train_error_curve.append(0, self.model.compute_mse(x_train, y_train))
        test_error_curve.append(0, self.model.compute_mse(x_test, y_test))

        self.train_progress_bar.setMaximum(epoch)
        self.train_progress_bar.setValue(0)

        for i, eval in self.model.train(x_train, y_train, epoch, LR_SCHEDULERS[lr_scheduler](learning_rate), momentum, num_classes, [ACCURACY], early_stop_acc):
            self.train_progress_bar.setValue(i+1)
            train_accuracy_curve.append(i+1, eval['accuracy'])
            test_accuracy_curve.append(i+1, self.model.evaluate(x_test, y_test, num_classes, metrics=[ACCURACY])['accuracy'])
            train_error_curve.append(i+1, self.model.compute_mse(x_train, y_train))
            test_error_curve.append(i+1, self.model.compute_mse(x_test, y_test))
        
        self.train_progress_bar.setValue(epoch)

        self.learn_chart.removeAllSeries()
        self.learn_chart_x_axis.setRange(0, i+1)
        if i >= 1:
            self.learn_chart_x_axis.setTickInterval(10**int(log10(i)))
        else:
            self.learn_chart_x_axis.setTickInterval(1)
        self.learn_chart.addSeries(train_accuracy_curve)
        self.learn_chart.addSeries(test_accuracy_curve)
        self.learn_chart.addSeries(train_error_curve)
        self.learn_chart.addSeries(test_error_curve)
        train_accuracy_curve.attachAxis(self.learn_chart_x_axis)
        train_accuracy_curve.attachAxis(self.learn_chart_y_axis)
        test_accuracy_curve.attachAxis(self.learn_chart_x_axis)
        test_accuracy_curve.attachAxis(self.learn_chart_y_axis)
        train_error_curve.attachAxis(self.learn_chart_x_axis)
        train_error_curve.attachAxis(self.learn_chart_y_axis)
        test_error_curve.attachAxis(self.learn_chart_x_axis)
        test_error_curve.attachAxis(self.learn_chart_y_axis)
        
        self.statusMessage.emit("Model training completed.")

        if standardize:
            p_1, p_2 = self.current_data_loader.standardized_bound_box
        else:
            p_1, p_2 = self.current_data_loader.bound_box

        x1 = p_1[0]-1
        x2 = p_2[0]+1
        
        self.data_chart.removeAllSeries()
        self.redraw_charts()

        if len(model_shape) == 2 and model_shape[0] == 2 and model_shape[1] == 1:
            weights = self.model.W[0][0].tolist()
            for i in range(num_classes-1):
                thr = (i+1)/num_classes
                model_line = lambda x: (weights[0]+self.model.activate_func.inverse(thr)-weights[1]*x)/weights[2]
                y1 = model_line(x1)
                y2 = model_line(x2)
                new_line_series = QLineSeries()
                new_line_series.setName("Decision boundary(thr={:.2f})".format(thr))
                new_line_series.append(x1, y1)
                new_line_series.append(x2, y2)
                self.data_chart.addSeries(new_line_series)
                new_line_series.attachAxis(self.data_chart_x_axis)
                new_line_series.attachAxis(self.data_chart_y_axis)
        

class MainWindow(QMainWindow):

    def __init__(self, widget: MainWidget):
        QMainWindow.__init__(self)
        self.widget = widget
        self.setCentralWidget(widget)
        self.setWindowTitle("Perceptron Demonstration")
        self.widget.statusMessage.connect(self.update_status_bar)

        # Menu
        self.menu = self.menuBar()
        self.action_menu = self.menu.addMenu("Action")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)

        train_action = QAction("Train model", self)
        train_action.setShortcut(QKeySequence("Ctrl+t"))
        train_action.triggered.connect(self.widget.train_model)

        self.action_menu.addAction(train_action)
        self.action_menu.addSeparator()
        self.action_menu.addAction(exit_action)

        # Status Bar
        self.status = self.statusBar()
        self.status.showMessage("No data is loaded yet.")

        # Window dimensions
        geometry = self.screen().availableGeometry()
        self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.8)

    def update_status_bar(self, message):
        self.status.showMessage(message)

if __name__ == "__main__":
    app = QApplication()
    window = MainWindow(MainWidget())
    window.show()
    app.exec()