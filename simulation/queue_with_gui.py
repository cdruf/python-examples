# -*- coding: utf-8 -*-
import sys
import time
from functools import reduce
from threading import Thread

import numpy as np
import pandas as pd
import simpy
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QToolBar

from util.step_function import StepFunction
from util.plotting import show_histogram

# %%
n_servers = 2
avg_inter_arrival_time = 5.0
avg_service_duration = 10.0
timeout_factor = 0.001


class Customer:
    next_id = 0
    times_spent_in_queue_till_service = []
    times_spent_in_service = []
    times_spent_in_system = []
    n_served = 0

    def __init__(self, env, queue, arrival_time):
        self.env = env
        self.queue = queue
        self.id = Customer.next_id
        Customer.next_id += 1
        self.arrival_time = arrival_time
        self.service_start_time = np.nan
        self.departure_time = np.nan
        print(f'{env.now:.4f}: arrival {self}')

        self.gets_served_event = env.event()

    def start_service(self):
        print(f'{self.env.now:.4f}: start service {self}')
        self.service_start_time = self.env.now
        Customer.times_spent_in_queue_till_service.append(self.env.now - self.arrival_time)

    def departure(self):
        print(f'{self.env.now:.4f}: departure {self}')
        self.departure_time = self.env.now
        Customer.times_spent_in_service.append(self.env.now - self.service_start_time)
        Customer.times_spent_in_system.append(self.env.now - self.arrival_time)
        Customer.n_served += 1

    def __str__(self):
        return f'Customer {self.id}'


class Server:
    next_id = 0

    def __init__(self, env):
        self.env = env
        self.id = Server.next_id
        Server.next_id += 1
        self.idle = True
        self.serve_start_times = []
        self.serve_end_times = []

    def serve(self, customer, queue):
        self.idle = False
        self.serve_start_times.append(self.env.now)
        service_duration = np.random.exponential(scale=avg_service_duration)
        print(f'{self.env.now:.4f}: {self} starts serving {customer} for {service_duration:.4f} periods')
        customer.start_service()

        time.sleep(service_duration * timeout_factor)
        yield self.env.timeout(service_duration)

        customer.departure()
        self.serve_end_times.append(self.env.now)
        self.idle = True

        queue.dispatch()

    def __str__(self):
        return f'Server {self.id}, idle={self.idle}'


class Queue:

    def __init__(self, env, servers: list, gui):
        self.env = env
        self.queue = []
        self.n_customers = 0
        self.n_customers_collector = StepFunction(xs=[0], ys=[])
        self.servers = servers
        self.gui = gui

    def add(self, customer: Customer):
        self.queue.append(customer)
        self.n_customers += 1
        self.n_customers_collector.append_step(self.env.now, self.n_customers)
        self.gui.set_queue_length(self.n_customers)
        print(f'{self.env.now:.4f}: {customer} entered queue, length {len(self)}')

        self.dispatch()

    def dispatch(self):
        server = self.get_idle_server()
        if self.n_customers > 0 and server is not None:
            customer = self.pop()
            customer.gets_served_event.succeed()
            self.env.process(server.serve(customer, self))

    def get_idle_server(self):
        for server in self.servers:
            if server.idle:
                return server
        return None

    def pop(self) -> Customer:
        customer = self.queue.pop(0)
        self.decrement_customers()
        print(f'{self.env.now:.4f}: {customer} left queue front, length {len(self)}')
        return customer

    def remove(self, customer):
        assert customer in self.queue
        position = self.queue.index(customer)
        self.queue.remove(customer)
        self.decrement_customers()
        print(f'{self.env.now:.4f}: {customer} left queue at position {position} of {len(self)}')

    def decrement_customers(self):
        self.n_customers -= 1
        assert self.n_customers >= 0
        self.n_customers_collector.append_step(self.env.now, self.n_customers)
        self.gui.set_queue_length(self.n_customers)

    def __len__(self):
        return self.n_customers

    def __str__(self):
        return f'Queue - length={len(self)}'


def arrival_process(env, queue: Queue, servers: list):
    while True:
        inter_arrival_time = np.random.exponential(scale=avg_inter_arrival_time)
        # countdown_thread = Thread(target=next_customer_countdown,
        #                           args=(gui, inter_arrival_time),
        #                           daemon=True)
        # countdown_thread.start()
        time.sleep(inter_arrival_time * timeout_factor)
        yield env.timeout(inter_arrival_time)
        customer = Customer(env, queue, env.now)
        queue.add(customer)


# %%

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Queue simulation")

        layout = QGridLayout()

        self.lb_queue_length_label = QLabel('Number of customers queueing')
        layout.addWidget(self.lb_queue_length_label, 0, 0)
        self.lb_queue_length_value = QLabel('0')
        layout.addWidget(self.lb_queue_length_value, 0, 1)

        self.lb_next_customer_countdown_label = QLabel('Next customer in')
        layout.addWidget(self.lb_next_customer_countdown_label, 1, 0)
        self.lb_next_customer_countdown_value = QLabel('0')
        layout.addWidget(self.lb_next_customer_countdown_value, 1, 1)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)

    def set_queue_length(self, n):
        self.lb_queue_length_value.setText(str(n))
        self.repaint()

    def set_time_till_next_arrival(self, n):
        self.lb_next_customer_countdown_value.setText(str(n))


# %%
"""
# Run simulation
"""


def run_simulation(gui, n_periods=500):
    env = simpy.Environment()
    servers = [Server(env) for i in range(n_servers)]
    queue = Queue(env, servers, gui)
    env.process(arrival_process(env, queue, servers))
    env.run(until=500)

    queue.n_customers_collector.plot("Queue length")

    time_in_queue = pd.Series(Customer.times_spent_in_queue_till_service, dtype=float)
    show_histogram(time_in_queue, 'Time spent in queue')

    time_in_service = pd.Series(Customer.times_spent_in_service)
    show_histogram(time_in_service, 'Time spent in service')

    print(f"N arrived: {Customer.next_id}")
    print(f"N served: {Customer.n_served}")
    print(f"N waiting: {len(queue)}")
    n_being_served = reduce(lambda x, y: x + int(not y.idle), servers, 0)
    print(f"N being served: {n_being_served}")


def next_customer_countdown(gui, secs):
    secs = round(secs)
    for i in range(secs, 0, -1):
        gui.set_time_till_next_arrival(i)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    simulation_thread = Thread(target=run_simulation, args=(gui,))
    simulation_thread.start()
    sys.exit(app.exec_())
