import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.font_manager

import numpy as np
from config import *
from utils import *
from UAV import UAV
from Bus import Bus
import scienceplots

import argparse

import matplotlib.patches as patches
import cmocean.cm as cmo
import seaborn as sns


# POI 위치 설정
X, Y, Z = 450, 700, 0
position = 0

RANDOM_TASK = [{'name': "low", 'min': 5, 'max': 10}, {'name': "medium", 'min': 10, 'max': 20},
               {'name': "large", 'min': 20, 'max': 50}]
SCHEME = ["Proposal", "Matching", "Offloading", "Local"]
#SIGMA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
POSITION = [1,2,3,4,5,6,7,8,9]
SIGMA = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
#POSITION = [9, 8, 7, 6, 5, 4, 3, 2, 1]
SIMUL_NAME = ["Bus Num", "UAV Num", "Budget", "Scheme", "CPU required", "Sigma", "Position"]
SAVE_X_NAME = ["Bus", "UAV", "Budget", "Scheme", "CPU required", "Sigma", "Posision"]
SAVE_Y_NAME = ["overhead", "UAV_utility", "bus_utility", "bus_num", "price"]

NUM_OBJECT = [NUM_BUS, NUM_UAV, BUDGET, len(SCHEME), len(RANDOM_TASK), 1, 9]
NUM_STEP = [NUM_BUS_STEP, NUM_UAV_STEP, NUM_BUDGET_STEP, len(SCHEME), len(RANDOM_TASK), len(SIGMA), len(POSITION)]
STEP = [BUS_STEP, UAV_STEP, BUDGET_STEP, 1, 1, 0.1, 1]
X_LABEL = ["Number of buses", "Number of UAVs", "Budget", "Scheme", "CPU required", "$\sigma$", "Position"]
Y_LABEL = ["UAV overhead", "UAV utility", "Bus utility", "Avg. # of offloaded buses per UAV", "CPU price"]
LEGEND_LABEL = ["Bus=", "UAV=", "Budget=", "", "", "Sigma=", "Position="]


def simul_value(type, i):
    if type == 0:
        return NUM_BUS - i * BUS_STEP
    elif type == 1:
        return NUM_UAV - i * UAV_STEP
    elif type == 2:
        return BUDGET - i * BUDGET_STEP
    elif type == 3:
        return SCHEME[i]
    elif type == 4:
        return RANDOM_TASK[i]
    elif type == 5:
        return SIGMA[i]
    elif type == 6:
        return POSITION[i]
    return -1


def mean_without_outliers(lst: list, decision):
    lst.sort()
    l = len(lst)
    l_min = int(l * 0.1)
    l_max = int(l * 0.9)
    # lst = lst[l_min:l_max]
    return round(Average(lst), decision)


if __name__ == "__main__":
    # parsing / default = UAV-Bus task
    parser = argparse.ArgumentParser(description="_")
    parser.add_argument("--x", type=int, default=5, help="x value in graph, Bus Num=0, UAV Num=1, Budget=2, Scheme=3, CPU=4, Sigma=5, Position=6")
    parser.add_argument("--y", type=int, default=3, help="legend in graph, Bus Num=0, UAV Num=1, Budget=2, Scheme=3, CPU=4, Sigma=5, Position=6")
    parser.add_argument("--real", type=bool, default=0, help="flag for using real data")
    parser.add_argument("--map", type=bool, default=0, help="flag for draw map")
    args = parser.parse_args()

    print("### SIMULATION START ###")
    paths = []
    simul_time = 0
    num_bus = NUM_BUS
    num_x_step = NUM_STEP[args.x]
    num_y_step = NUM_STEP[args.y]

    if args.real:
        if args.x == 0:
            num_x_step = 1
        if args.y == 0:
            num_y_step = 1
        simul_time = 10

        with open("./buspos.txt", "r") as fp:
            t = 0
            while t < simul_time:
                path = []
                line = fp.readline()
                poslst = line.split('/')[:-1]
                for pos in poslst:
                    x, y = np.array(pos.split(','), dtype=np.float32)
                    path.append((x, y))

                paths.append(path)
                t += 1

        paths = np.array(paths).transpose((1, 0, 2))

        num_bus = len(paths)
    else:
        # make environment
        simul_time = SIMUL_TIME
        for i in range(num_bus):
            path = [(random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE))]
            while len(path) < NUM_PATH:
                x, y = path[-1]
                next_x = random.randint(max(0, x - random.randint(1, 50)), min(MAP_SIZE, x + random.randint(1, 50)))
                next_y = random.randint(max(0, y - random.randint(1, 50)), min(MAP_SIZE, y + random.randint(1, 50)))
                if math.dist((x, y), (next_x, next_y)) >= 50:
                    path.append((next_x, next_y))
            paths.append(path)

    buses_original = []
    for i in range(num_bus):
        buses_original.append(Bus(i, 0, paths[i]))

    if args.map:
        draw_map(X,Y, buses_original)

    # POI로부터 일정거리 이내에 위치하도록 UAV 생성

    uavs_original = []
    for i in range(NUM_UAV):
        uavs_original.append(UAV(i, X, Y, Z, position))

    # for graph
    uav_bus_avg_overhead = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    uav_avg_utility = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    bus_avg_utility = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    uav_avg_bus_num = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    bus_avg_price = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]

    buses = deepcopy(buses_original)
    uavs = deepcopy(uavs_original)
    scheme = SCHEME[0]
    budget = BUDGET
    sigma = ALPHA

    task_range = {'min': TASK_CPU_CYCLE, 'max': TASK_CPU_CYCLE}
    # UAV 대수 - 버스의 대수를 점점 줄여나가면서 시뮬레이션
    for i in range(num_y_step):
        if args.y == 3:
            scheme = SCHEME[i]
        elif args.y == 4:
            task_range = RANDOM_TASK[i]


        if args.x == 0:
            buses = deepcopy(buses_original)
        elif args.x == 1:
            uavs = deepcopy(uavs_original)
        elif args.x == 2:
            budget = BUDGET
        elif args.x == 3:
            pass
        elif args.x == 5:
            sigma = SIGMA[i]


        for j in range(num_x_step):
            print(f"### {SIMUL_NAME[args.y]}:{simul_value(args.y, i)} {SIMUL_NAME[args.x]}:{simul_value(args.x, j)} simulation start")

            if args.x == 6:
                position = POSITION[j]
                #print(position)
                uavs = []
                for k in range(NUM_UAV):
                    uavs.append(UAV(k, X, Y, Z, position))

            # simulate
            simulation(simul_time, uavs, buses, scheme=scheme, budget=budget, random_task_range=task_range, sigma=sigma)
            print("### SIMULATION RESULT ###")

            tmp_overhead = []
            tmp_uav_utility = []
            tmp_bus_utility = []
            tmp_bus_num = []
            tmp_bus_avg_price = []

            under = 0
            upper = 0

            for uav in uavs:
                #print(uav.overhead_list)
                if len(uav.overhead_list) > 0:
                    avg_overhead = mean_without_outliers(uav.overhead_list, 4)
                    tmp_overhead.append(avg_overhead)
                if len(uav.utility_list) > 0:
                    avg_utility = mean_without_outliers(uav.utility_list, 4)
                    tmp_uav_utility.append(avg_utility)
                if len(uav.bus_num_list) > 0:
                    avg_bus_num = mean_without_outliers(uav.bus_num_list, 4)
                    tmp_bus_num.append(avg_bus_num)
                if len(uav.price_list) > 0:
                    avg_price = round(Average(uav.price_list), 4)
                    tmp_bus_avg_price.append(avg_price)

                #print(f"UAV(ID={uav.id}) has overhead")  # : {avg_overhead}, utility : {avg_utility}, price : {avg_price}")

            for bus in buses:
                if len(bus.utility_list) > 0:
                    avg_utility = mean_without_outliers(bus.utility_list, 4)
                    # print(bus.price_list)
                    tmp_bus_utility.append(avg_utility)
                #print(f"BUS(ID={bus.id}) has utility ")  #: {avg_utility}")

            uav_bus_avg_overhead[i][j] = round(Average(tmp_overhead), 4)  # mean_without_outliers(tmp_overhead,4)
            uav_avg_utility[i][j] = round(Average(tmp_uav_utility), 4)  # mean_without_outliers(tmp_uav_utility,4)
            uav_avg_bus_num[i][j] = round(Average(tmp_bus_num), 4)  # mean_without_outliers(tmp_bus_num,4)
            bus_avg_utility[i][j] = round(Average(tmp_bus_utility), 4)  # mean_without_outliers(tmp_bus_utility,4)
            bus_avg_price[i][j] = round(Average(tmp_bus_avg_price), 4)

            # print(f"UAV overhead : {AVE}, Under : {under}, Upper : {upper}")
            if args.x == 0:
                for k in range(BUS_STEP):
                    del buses[-1]
            elif args.x == 1:
                for k in range(UAV_STEP):
                    del uavs[-1]
            elif args.x == 2:
                budget -= BUDGET_STEP
            elif args.x == 3:
                scheme = SCHEME[j]
            elif args.x == 4:
                task_range = RANDOM_TASK[j]
            elif args.x == 5:
                sigma = SIGMA[j]
            elif args.x == 6:
                position = POSITION[j]

        if args.y == 0:
            for k in range(BUS_STEP):
                del buses[-1]
        elif args.y == 1:
            for k in range(UAV_STEP):
                del uavs[-1]
        elif args.y == 2:
            budget -= BUDGET_STEP

    # print result

    x_idx = np.arange(0, num_x_step)
    x = NUM_OBJECT[args.x] - x_idx * STEP[args.x]
    legend_value_idx = np.arange(0, num_y_step)
    legend_value = []
    for i in legend_value_idx:
        v = NUM_OBJECT[args.y] - i * STEP[args.y]
        if args.y == 3:
            v = SCHEME[NUM_OBJECT[args.y] - v]
        elif args.y == 4:
            v = RANDOM_TASK[NUM_OBJECT[args.y] - v]['name']
        legend_value.append(v)
    data = [uav_bus_avg_overhead, uav_avg_utility, bus_avg_utility, uav_avg_bus_num, bus_avg_price]

    marker = itertools.cycle(('+', '2', '.', 'x', '*'))
    plt.style.use(['science', 'ieee', 'no-latex'])

    for i in range(5):

        data2 = []

        for j in range(len(legend_value)):

            if args.x != 6:
                plt.plot(x, data[i][j], marker = next(marker), label=LEGEND_LABEL[args.y] + str(legend_value[j]))
                plt.xticks(x)

            #if str(legend_value[j])=="Proposal":
            #plt.bar(x, data[i][j], label=LEGEND_LABEL[args.y] + str(legend_value[j]))
            #plt.xticks(x)

            data2.append(data[i][j])

        if args.x == 6:
            width = 0.15  # the width of the bars
            multiplier = 0
            for k in range(len(data2)):
                offset = width * multiplier
                rects = plt.bar(x + offset, data2[k], width, label=LEGEND_LABEL[args.y] + str(legend_value[k]))
                multiplier += 1
                plt.xticks(x)

        plt.xlabel(X_LABEL[args.x])
        plt.ylabel(Y_LABEL[i])
        plt.legend(loc='best')
        plt.legend(frameon=True)
        plt.savefig("./test_graphs/" + SAVE_X_NAME[args.x] + "_" + SAVE_X_NAME[args.y] + "_" + SAVE_Y_NAME[i])
        plt.clf()

    # marker = itertools.cycle(('+', '2', '.', 'x'))
    #
    # plt.plot(x, data[0][0], marker = next(marker), label="UAV overhead")
    # plt.plot(x, data[1][0], marker = next(marker), label="UAV utility")
    # plt.plot(x, data[2][0], marker = next(marker), label="Bus utility")
    #
    # plt.xlabel(X_LABEL[args.x])
    # plt.ylabel("overhead,utility")
    # plt.legend(loc='best')
    # plt.legend(frameon=True)
    # plt.savefig("./test_graphs/" + SAVE_X_NAME[args.x] + "_" + SAVE_X_NAME[args.y] + "_" + "overutil")