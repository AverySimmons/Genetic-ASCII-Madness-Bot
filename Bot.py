import pyautogui as agui
import numpy as np
import ComputerVision
import PyGadLearning
import keyboard
import time

def main():
    running = False

    prev_frame = None
    prev_player_pos = np.array([100,100])
    prev_cursor_pos = []
    cursor_num = 4
    rec_num = 1
    rec_node_values = [0 for _ in range(rec_num)]
    filepath = "Replays/full/gen_47/pop_182/solution.npy"
    model = PyGadLearning.loadModel(filepath)
    print("model ready (press n to start, q to stop)")
    while True:
        if running:
            start_time = time.time()
            inp, close_code, prev_cursor_pos, prev_player_pos, prev_frame = ComputerVision.getNetworkInput(
                cursor_num, prev_frame, prev_cursor_pos, prev_player_pos)
            out = PyGadLearning.calcModel(model, inp, rec_node_values)
            rec_node_values = out[len(out) - rec_num:]
            proccessOutput(out)
            autoAttack(close_code)
            
            if keyboard.is_pressed("q"):
                print("bot toggled off!")
                resetKeys()
                running = False
        else:
            time.sleep(0.03)

            if keyboard.is_pressed("n"):
                print("bot toggled on!")
                running = True

def proccessOutput(out):
    resetKeys()
    if out[0] > 0:
        keyboard.press("w")
    if out[1] > 0:
        keyboard.press("s")
    if out[2] > 0:
        keyboard.press("d")
    if out[3] > 0:
        keyboard.press("a")
    if out[4] > 0:
        keyboard.press("space")

def resetKeys():
    keyboard.release("w")
    keyboard.release("s")
    keyboard.release("d")
    keyboard.release("a")
    keyboard.release("space")

def autoAttack(pos):
    pos[0] += 50
    pos[1] += 95
    agui.click(pos[0], pos[1])

def normalize(vect):
    norm = np.linalg.norm(vect)
    if norm == 0: 
       return vect
    return vect / norm

if __name__ == "__main__":
    main()