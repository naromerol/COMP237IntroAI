# -*- coding: utf-8 -*-
"""
COMP237 - Introduction to AI
Search Assignment - Ex1
@author: NestorRomero
"""

#graph to store the relationships
graph = {}
graph["Adam"] = {"Ema", "Nestor", "Bob"}
graph["Ema"] = {"Adam", "Bob", "Dolly"}
graph["Nestor"] = {"Adam", "George", "Frank"}
graph["Bob"] = {"Adam", "Ema", "Dolly"}
graph["George"] = {"Nestor"}
graph["Frank"] = {"Nestor"}
graph["Dolly"] = {"Bob", "Ema"}