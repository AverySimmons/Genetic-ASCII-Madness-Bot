
# Genetic ASCII Madness Bot

---

**Tools Used:** Python, PyGad, cv2, Tensorflow &nbsp;&nbsp;&nbsp;&nbsp; **Keywords:** Computer vision, Genetic learning, Optimization

---

Note: ASSCI Madness is not my project - check out the original work [here](https://github.com/big-evil-fish/ASCII-MADNESS-GAME-)

### Description:
&nbsp;&nbsp;&nbsp;&nbsp;ASCII Madness is a bullet hell roguelike made by a friend of mine. I had always wanted to make a bot to play a game, and here was the perfect opportunity; the game was fun and skill-based, with simple enough graphics for basic computer vision to work. More importantly, the source code was available, meaning that I would not have to recreate the game from scratch to train an AI.
<br><br>
First, I removed all source code related to displaying graphics. Since ASCII Madness is built using the CMU Graphics library, this also meant reorganizing game logic to eliminate dependencies on certain built-in functions. Next, I combined the Python libraries PyGad and TensorFlow to train a neural network to play the game with proficiency. To aid troubleshooting during this step, I also created a replay system that saved the coordinates of objects in the played game for later playback.
<br><br>
Finally, I used the Python library cv2 to create custom visual recognition for various game elements and hooked them up as inputs for the trained neural network.


### Features:
- &nbsp;&nbsp;&nbsp;&nbsp;**AI Parameters:**  
The current rendition of the model is given 15 inputs: x/y from center of screen, angle to closest code block, and the local position of the closest 4 cursors in polar cordinates as well as the local angle of their velocities. For output the network has access to the wasd keys - which are bound to the directionial movement - as well as the dash key (space). The best model I have found was trained using a large population with a high mutation for around 100 generations.

- &nbsp;&nbsp;&nbsp;&nbsp;**Optimizing Training Speeds:**  
The primary time sink of this project was waiting for new models to be trained as a population of hundreds with a low convergence was needed in order to aptly explore the sample space. In order to more efficently expirement with new parameters and methods, optimizing the training of networks was a must. To this end - as mentioned in the description - I removed all source code using the CMU Graphics library and instead created a simple replay system that wrote objects positions to files as games where played. Furthermore I also used the parallel processing settings with PyGad to have the fitness of the population be assessed using multiple proccesses.

- &nbsp;&nbsp;&nbsp;&nbsp;**Computer Vision:**  
Objects on screen are recognized by color. The main roadblock with this technique is that many game sprites are the exact same color. For example pieces of the player's body can be either white, which is the same color as cursors, or red, which is the same color as code blocks. For cursors the fix was simple enough, using the cv2 library I simply sorted large white objects as cursors and small ones as player segments. For issues with red objects I needed a more complex solution as both code blocks and the player body pieces are the same size. To fix this I stored every previus frame and used it to check which objects where in motion (or rather whether the color in a particular region had changed since the last frame), this way a red region could be identified as a player piece if in motion and a code block if not. This previus frame technique was also used to calculate the direction of the cursors velocity.


### Code Breakdown:
- &nbsp;&nbsp;&nbsp;&nbsp;**Game:**  
The game file contains all the logic of ASCII Madness with some modifications as well as additional function to calculate inputs for the AI model. (The vast majority of code in this file is **not mine**)

- &nbsp;&nbsp;&nbsp;&nbsp;**PyGad Learning:**  
This file contains all the logic for training the AI, including but not limited to PyGad logic to run a genetic algorithm on a population of random arrays and Tensor Flow functions to take those arrays, convert them to neural networks, and access their fitness.

- &nbsp;&nbsp;&nbsp;&nbsp;**Computer Vision:**  
This is the file that holds all code for image recognition using cv2.

- &nbsp;&nbsp;&nbsp;&nbsp;**Bot:**  
This file links PyGad learning and computer vision to run the bot.
