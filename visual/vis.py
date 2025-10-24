import matplotlib.pyplot as plt
from dataset import GestureDataset

# 0 none 1 wave_right 2 wave_down 3 wave_left 4 wave_up 5 tap_air
# 6 tap_plane 7 push_forward 8 pinch 9 clench 10 flip 11 wrist_clockwise
# 12 wrist_counterclockwise 13 circle_clockwise 14 circle_counterclockwise 15 clap 16 snap
# 17 thumb_up 18 middle_pinch 19 index_flick 20 touch_plane 21 thumb_tap_index
# 22 index_bend_and_straighten 23 ring_pinch 24 pinky_pinch 25 slide_plane 26 pinch_down
# 27 pinch_up 28 boom 29 tap_up 30 throw 31 touch_left 32 touch_right 33 slide_up
# 34 slide_down 35 slide_left 36 slide_right 37 aid_slide_left 38 aid_slide_right 39 touch_up
# 40 touch_down 41 touch_ring 42 long_touch_ring 43 spread_ring

def visual(x):
    # fig, ax = plt.subplots(figsize=(12, 6))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(x[i], label=f'Line {i+1}')
    plt.title('Line Plots for Rows 1-3')
    plt.xlabel('Index (Columns)')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(3, 6):
        plt.plot(x[i], label=f'Line {i+4}')
    plt.title('Line Plots for Rows 4-6')
    plt.xlabel('Index (Columns)')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # for i in range(x.shape[0]):
    #     ax.plot(x[i], label=f'Line {i+1}')

    # ax.set_title('Line Plots for 6x200 Array')
    # ax.set_xlabel('Index (Columns)')
    # ax.set_ylabel('Value')
    # ax.legend()
    # plt.show()

dataset = GestureDataset(valid = [0, 5, 6, 8, 16])

for x, y in dataset:
    if y == 3:
        visual(x)
