import sys
import math
import numpy as np
from scipy.linalg import eigh
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem,
    QDialog, QLabel, QLineEdit, QFormLayout, QMessageBox, QGraphicsTextItem,
    QListWidget, QListWidgetItem, QComboBox, QSpacerItem, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter


class Node(QGraphicsEllipseItem):
    def __init__(self, x, y, name, radius=20):
        super().__init__(-radius, -radius, 2*radius, 2*radius)
        self.setBrush(QBrush(QColor("skyblue")))
        self.setPen(QPen(Qt.black, 2))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable)
        self.setPos(x, y)
        # 节点名称
        self.name = name
        # 默认参数
        self.properties = {
            'P_i': 1.0,
            'alpha': 0.1,
            'K_ij': {}
        }
        self.text = None  # 用于显示DVI值
        self.name_text = QGraphicsTextItem(self.name, self)
        self.name_text.setDefaultTextColor(Qt.black)
        self.name_text.setPos(-radius, radius + 5)  # 相对于节点的位置

    def contextMenuEvent(self, event):
        dialog = ParameterDialog(self)
        dialog.exec_()


class Edge(QGraphicsLineItem):
    def __init__(self, node1, node2):
        super().__init__()
        self.node1 = node1
        self.node2 = node2
        self.setPen(QPen(Qt.black, 2))
        self.update_position()

    def update_position(self):
        pos1 = self.node1.pos()
        pos2 = self.node2.pos()
        self.setLine(pos1.x(), pos1.y(), pos2.x(), pos2.y())


class ParameterDialog(QDialog):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle("编辑节点参数")
        self.setFixedSize(300, 150)
        layout = QFormLayout()

        self.p_input = QLineEdit(str(self.node.properties['P_i']))
        self.alpha_input = QLineEdit(str(self.node.properties['alpha']))

        layout.addRow("P_i:", self.p_input)
        layout.addRow("alpha:", self.alpha_input)

        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self.save_parameters)
        layout.addWidget(btn_save)

        self.setLayout(layout)

    def save_parameters(self):
        try:
            p = float(self.p_input.text())
            alpha = float(self.alpha_input.text())
            self.node.properties['P_i'] = p
            self.node.properties['alpha'] = alpha
            QMessageBox.information(self, "成功", "参数已保存。")
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数值。")


class NetworkGraph(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.edges = []
        self.mode = None  # 'add', 'remove', 'link', None
        self.temp_link = []  # 临时存储要链接的两个节点
        self.nodes_changed_callback = None  # 回调函数，当节点变化时调用
        self.node_count = 0  # 记录节点数量，用于命名

    def set_mode(self, mode):
        self.mode = mode
        self.temp_link = []

    def add_node(self, x, y):
        self.node_count += 1
        node_name = f"节点 {self.node_count}"
        node = Node(x, y, node_name)
        self.addItem(node)
        self.nodes.append(node)
        if self.nodes_changed_callback:
            self.nodes_changed_callback()

    def remove_node(self, node):
        # 移除与该节点相关的所有边
        edges_to_remove = [edge for edge in self.edges if edge.node1 == node or edge.node2 == node]
        for edge in edges_to_remove:
            self.removeItem(edge)
            self.edges.remove(edge)
        self.removeItem(node)
        self.nodes.remove(node)
        if self.nodes_changed_callback:
            self.nodes_changed_callback()

    def add_edge(self, node1, node2):
        # 检查是否已经存在连接
        for edge in self.edges:
            if (edge.node1 == node1 and edge.node2 == node2) or (edge.node1 == node2 and edge.node2 == node1):
                QMessageBox.warning(None, "警告", "这两个节点已经连接。")
                return
        edge = Edge(node1, node2)
        self.addItem(edge)
        self.edges.append(edge)
        # 更新耦合强度K_ij
        node1.properties['K_ij'][node2] = 1.0  # 默认K_ij=1.0
        node2.properties['K_ij'][node1] = 1.0

    def mousePressEvent(self, event):
        if self.mode == 'add':
            pos = event.scenePos()
            self.add_node(pos.x(), pos.y())
        elif self.mode == 'remove':
            items = self.items(event.scenePos())
            for item in items:
                if isinstance(item, Node):
                    self.remove_node(item)
                    break
        elif self.mode == 'link':
            items = self.items(event.scenePos())
            for item in items:
                if isinstance(item, Node):
                    if item not in self.temp_link:
                        self.temp_link.append(item)
                        if len(self.temp_link) == 2:
                            self.add_edge(self.temp_link[0], self.temp_link[1])
                            self.temp_link = []
                    break
        else:
            super().mousePressEvent(event)

    def update_edges(self):
        for edge in self.edges:
            edge.update_position()


class GraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.zoom_factor = 1.15

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DVI & Lyapunov 可视化模拟")
        self.setGeometry(100, 100, 1800, 1000)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧菜单栏
        menu_layout = QVBoxLayout()

        # 驱动节点选择
        self.driver_list = QListWidget()
        self.driver_list.setSelectionMode(QListWidget.MultiSelection)
        menu_layout.addWidget(QLabel("选择驱动节点:"))
        menu_layout.addWidget(self.driver_list)

        # 功率谱密度选择
        self.s_omega_combo = QComboBox()
        self.s_omega_combo.addItem("白噪声 S(ω) = 1")
        self.s_omega_combo.addItem("自定义 S(ω)")
        self.s_omega_combo.currentIndexChanged.connect(self.s_omega_selection_changed)

        self.s_omega_input = QLineEdit()
        self.s_omega_input.setPlaceholderText("请输入 S(ω) 的值")
        self.s_omega_input.setEnabled(False)  # 初始禁用

        menu_layout.addWidget(QLabel("功率谱密度 S(ω):"))
        menu_layout.addWidget(self.s_omega_combo)
        menu_layout.addWidget(self.s_omega_input)

        # 操作按钮
        btn_add = QPushButton("增加节点")
        btn_remove = QPushButton("减少节点")
        btn_link = QPushButton("链接节点")
        btn_compute = QPushButton("计算DVI")

        btn_add.clicked.connect(self.add_node_mode)
        btn_remove.clicked.connect(self.remove_node_mode)
        btn_link.clicked.connect(self.link_nodes_mode)
        btn_compute.clicked.connect(self.compute_dvi)

        menu_layout.addWidget(btn_add)
        menu_layout.addWidget(btn_remove)
        menu_layout.addWidget(btn_link)
        menu_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        menu_layout.addWidget(btn_compute)

        # 添加DVI结果展示表格
        self.dvi_table = QTableWidget()
        self.dvi_table.setColumnCount(0)
        self.dvi_table.setRowCount(0)
        self.dvi_table.setVisible(False)  # 初始隐藏
        menu_layout.addWidget(QLabel("DVI 结果 (驱动节点 -> 目标节点):"))
        menu_layout.addWidget(self.dvi_table)

        # 画布
        self.scene = NetworkGraph()
        self.view = GraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        self.view.setMouseTracking(True)

        # 连接节点移动事件以更新边的位置
        self.scene.installEventFilter(self)

        # 设置节点变化回调
        self.scene.nodes_changed_callback = self.update_driver_list

        main_layout.addLayout(menu_layout)
        main_layout.addWidget(self.view)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def add_node_mode(self):
        self.scene.set_mode('add')
        self.statusBar().showMessage("模式: 增加节点")

    def remove_node_mode(self):
        self.scene.set_mode('remove')
        self.statusBar().showMessage("模式: 减少节点")

    def link_nodes_mode(self):
        self.scene.set_mode('link')
        self.statusBar().showMessage("模式: 链接节点")

    def update_driver_list(self):
        self.driver_list.clear()
        for idx, node in enumerate(self.scene.nodes):
            item = QListWidgetItem(node.name)
            item.setData(Qt.UserRole, node)
            item.setCheckState(Qt.Unchecked)
            self.driver_list.addItem(item)

    def s_omega_selection_changed(self, index):
        if index == 1:  # 自定义
            self.s_omega_input.setEnabled(True)
        else:
            self.s_omega_input.setEnabled(False)

    def compute_dvi(self):
        selected_items = [
            self.driver_list.item(i)
            for i in range(self.driver_list.count())
            if self.driver_list.item(i).checkState() == Qt.Checked
        ]

        if len(selected_items) == 0:
            QMessageBox.warning(self, "警告", "请选择至少一个驱动节点。")
            return

        # 获取驱动节点
        driver_nodes = [item.data(Qt.UserRole) for item in selected_items]

        N = len(self.scene.nodes)
        if N < 2:
            QMessageBox.warning(self, "警告", "网络至少需要两个节点才能计算DVI。")
            return

        # 创建节点索引映射
        node_indices = {node: idx for idx, node in enumerate(self.scene.nodes)}

        # 构建拉普拉斯矩阵
        L = np.zeros((N, N))
        for i, node in enumerate(self.scene.nodes):
            for neighbor in node.properties['K_ij']:
                j = node_indices[neighbor]
                K_ij = node.properties['K_ij'][neighbor]
                L[i][i] += K_ij
                L[i][j] -= K_ij

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = eigh(L)

        # 检查是否有足够的特征值
        if N < 2:
            QMessageBox.warning(self, "警告", "网络至少需要两个节点才能计算DVI。")
            return

        # 获取第一个非零特征值和特征向量（索引1）
        lambda1 = eigenvalues[1]
        v1 = eigenvectors[:, 1]

        # 获取全局alpha（取所有节点alpha的平均值）
        alphas = [node.properties['alpha'] for node in self.scene.nodes]
        global_alpha = sum(alphas) / len(alphas)

        # 获取S(omega)
        s_omega_choice = self.s_omega_combo.currentText()
        if s_omega_choice == "白噪声 S(ω) = 1":
            S_omega = lambda omega: 1.0
        elif s_omega_choice == "自定义 S(ω)":
            try:
                S_val = float(self.s_omega_input.text())
                S_omega = lambda omega: S_val
            except ValueError:
                QMessageBox.warning(self, "错误", "请输入有效的 S(ω) 值。")
                return
        else:
            S_omega = lambda omega: 1.0

        # 计算共振频率
        if lambda1 <= (global_alpha ** 2) / 4:
            QMessageBox.warning(self, "警告", "第一个非零特征值不满足共振条件 (λ1 > α² / 4)。")
            return

        omega_res = math.sqrt(lambda1 - (global_alpha ** 2) / 4)

        # 定义共振频率区间 I_res
        delta = 0.1 * omega_res  # 10% 的共振频率作为区间宽度
        omega_min = max(0.0, omega_res - delta)
        omega_max = omega_res + delta
        delta_omega = 0.01  # 频率步长
        omegas = np.arange(omega_min, omega_max, delta_omega)

        # 初始化DVI矩阵
        DVI_matrix = np.zeros((len(driver_nodes), N))

        # 进行DVI计算
        for k, driver_node in enumerate(driver_nodes):
            driver_index = self.scene.nodes.index(driver_node)
            v_k = v1[driver_index]
            for i in range(N):
                v_i = v1[i]
                abs_v_k_v_i = np.abs(v_k * v_i)
                # 计算分母 sqrt((omega^2 - lambda1)^2 + (alpha * omega)^2)
                denominator = np.sqrt((omegas**2 - lambda1)**2 + (global_alpha * omegas)**2)
                # 避免除以零
                denominator = np.where(denominator == 0, 1e-10, denominator)
                # 计算被积函数
                integrand = S_omega(omegas) * omegas * abs_v_k_v_i / denominator
                # 使用梯形法则进行数值积分
                DVI_contribution = np.trapz(integrand, omegas)
                # 存储到DVI矩阵
                DVI_matrix[k][i] = DVI_contribution

        # 确保DVI值非负
        DVI_matrix = np.abs(DVI_matrix)

        # 更新DVI结果表格
        self.update_dvi_table(driver_nodes, DVI_matrix)

        # 可选：累计所有驱动节点的DVI值用于可视化
        cumulative_dvi = DVI_matrix.sum(axis=0)

        # 归一化累计DVI值用于可视化（可选）
        max_dvi = np.max(cumulative_dvi)
        if max_dvi == 0:
            max_dvi = 1.0
        normalized_cumulative_dvi = cumulative_dvi / max_dvi  # Scale between 0 and 1

        # 显示累计DVI值
        for idx, node in enumerate(self.scene.nodes):
            DVI = cumulative_dvi[idx]
            if node.text:
                self.scene.removeItem(node.text)
            # 使用颜色渐变表示DVI值（红色越高脆弱性越大）
            color_intensity = min(int(normalized_cumulative_dvi[idx] * 255), 255)
            color = QColor(255, 255 - color_intensity, 255 - color_intensity)
            node.setBrush(QBrush(color))

            # 调整节点大小基于DVI值
            base_radius = 20
            new_radius = base_radius + int(normalized_cumulative_dvi[idx] * 20)  # 半径在20到40之间
            node.setRect(-new_radius, -new_radius, 2*new_radius, 2*new_radius)

            # 添加累计DVI文本
            node.text = QGraphicsTextItem(f"DVI: {DVI:.2f}", node)
            node.text.setDefaultTextColor(Qt.red)
            node.text.setPos(new_radius + 5, -new_radius - 15)  # 相对于节点的位置

        QMessageBox.information(self, "完成", "DVI计算完成。")

    def update_dvi_table(self, driver_nodes, DVI_matrix):
        num_drivers = len(driver_nodes)
        num_targets = len(self.scene.nodes)
        self.dvi_table.setRowCount(num_drivers)
        self.dvi_table.setColumnCount(num_targets + 1)  # +1 for driver node name
        headers = ["驱动节点"] + [f"节点 {i + 1}" for i in range(num_targets)]
        self.dvi_table.setHorizontalHeaderLabels(headers)
        self.dvi_table.setVerticalHeaderLabels([f"驱动 {k + 1}" for k in range(num_drivers)])

        for k in range(num_drivers):
            driver_name = driver_nodes[k].name
            item = QTableWidgetItem(driver_name)
            self.dvi_table.setItem(k, 0, item)
            for i in range(num_targets):
                dvi_value = DVI_matrix[k][i]
                item = QTableWidgetItem(f"{dvi_value:.2f}")
                self.dvi_table.setItem(k, i + 1, item)

        self.dvi_table.resizeColumnsToContents()
        self.dvi_table.resizeRowsToContents()
        self.dvi_table.setVisible(True)
        self.dvi_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dvi_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
