# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:39:06 2020

@author: 49856
"""
import sys
from abc import ABC, abstractmethod 

import heapq

import numpy as np
import pandas as pd



#%% 
## Basic classes

class Event(ABC):
    
    def __init__(self, time, scheduled_at=0, name=''):
        self.time = time
        self.scheduled_at = scheduled_at
        self.name = name
    
    # abstract method
    def run(self):
        pass
        
    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time
    
    def __str__(self):
        name_str = ' ' + self.name if self.name != '' else ''
        return 'Event%s, time %d, scheduled at %d' % (name_str, self.time, self.scheduled_at)

class Env():
    
    def __init__(self):
        self.now = 0
        self.heap = [] # event queue 
        # heap => obtain next event in O(1), insert new event in O(log(n)) 

    def push(self, event):
        assert event.time >= self.now
        heapq.heappush(self.heap, event)
    
    def sim(self, until = sys.maxsize):
        while len(self.heap) > 0 and self.now <= until:
            event = heapq.heappop(self.heap)
            self.now = event.time
            event.run()

#%%
## Basic example
    
class Event1(Event):
    def run(self):
        print(self)

env = Env()
env.push(Event1(5, scheduled_at=0, name='Explosion'))
env.push(Event1(7))
env.push(Event1(1, name='Heist'))
env.push(Event1(1))
#env.push(Event(3,params=['param1']))

env.sim()

#%%
class Event2(Event):
    
    def __init__(self, time, greeting, scheduled_at=0, name=''):
        super().__init__(time, scheduled_at=scheduled_at, name=name)
        self.greeting = greeting
    
    def run(self):
        print(self)
        print('%s' % self.greeting)

env = Env()
env.push(Event2(5, 'hi', scheduled_at=0, name='Explosion'))
env.push(Event2(7, 'hello'))
env.push(Event2(1, 'hallo', name='Heist'))
env.push(Event2(1, 'moin'))

env.sim()


#%% 
# Events created by events

class Event3(Event):
    
    number = 0
    
    def run(self):
        Event3.number += 1
        print(str(Event3.number), ':', str(self))
        t = env.now + np.random.geometric(0.3)
        env.push(Event3(t, scheduled_at=env.now))

env = Env()
env.push(Event3(0))
env.sim(until=100)

#%%
# Car example

# alternatingly driving and parking

class EventPark(Event):
    def run(self):
        print(self)
        print('Start parking at %d' % env.now)
        duration = 5
        env.push(EventDrive(env.now+duration, scheduled_at=env.now, name='Drive'))

class EventDrive(Event):
    def run(self):
        print(self)
        print('Start driving at %d' % env.now)
        duration = 2
        env.push(EventPark(env.now+duration, scheduled_at=env.now, name='Park'))


env = Env()
env.push(EventPark(0, scheduled_at=0, name='Start'))
env.push(Event1(3, scheduled_at=0, name='Roadrunner crossed the street'))
env.sim(until=20)


#%% 
## Queue example

queue = []
idle = True

avg_inter_arrival_time = 5
avg_service_duration = 7

waiting_times = []

class Customer:
    def __init__(self, name, arrival_time):
        self.name = name
        self.arrival_time = arrival_time
    def departure(self, departure_time):
        self.departure_time = departure_time
        waiting_times.append(departure_time-self.arrival_time)
    def __str__(self):
        return self.name

class EventCustomerArrival(Event):
    number = 0
    def run(self):
        EventCustomerArrival.number += 1
        customer = Customer('c_%d' % EventCustomerArrival.number, env.now)
        print('t_%d: Arrival %s' % (env.now, customer))
        queue.append(customer)
        print('Queue length is %d' % len(queue))

        # Schedule next customer arrival
        inter_arrival_time = np.random.geometric(1.0/avg_inter_arrival_time)
        print('Next arrival in %d' % inter_arrival_time)
        env.push(EventCustomerArrival(env.now + inter_arrival_time, scheduled_at=env.now))
        
        # Activate servicing if necessary
        global idle
        if idle:
            print('t_%d: Start servicing ' % (env.now))
            idle = False
            service_duration = np.random.geometric(1.0/avg_service_duration)
            print('Service duration is (%d)' % (service_duration))
            env.push(EventCustomerDeparture(env.now+service_duration, scheduled_at=env.now))

class EventCustomerDeparture(Event):
    def run(self):
        customer = queue.pop(0)
        customer.departure(env.now)
        print('t_%d: Departure %s' % (env.now, customer))
        print('Queue length is %d' % len(queue))

        if len(queue) == 0:
            global idle
            idle = True
            print('Queue empty - going to sleep')
        else:
            # Keep servicing waiting customers
            service_duration = np.random.geometric(1.0/avg_service_duration)
            print('Service duration is (%d)' % (service_duration))
            env.push(EventCustomerDeparture(env.now+service_duration, scheduled_at=env.now))

        

env = Env()
env.push(EventCustomerArrival(0, scheduled_at=0, name='First customer'))
env.sim(until=200)
print()
print('Final queue length is %d' % len(queue))
print(pd.Series(waiting_times).describe())


#%%
# Maintenance example
