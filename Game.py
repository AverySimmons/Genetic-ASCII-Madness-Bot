#credit to 112 tetris for a few matrix collision functions!
import copy
import math
import random
import pygame as py
import numpy as np

###### CLASSES

class Replay:
    def __init__(self) -> None:
        self.player_pos = []
        self.cursor_poses = []
        self.code_block_poses = []
        self.fitness = 0
    
    def capture(self, app):
        self.player_pos.append(py.Vector2(app.player.cx, app.player.cy))

        new_cur_poses = []
        for cur in app.cursors:
            new_cur_poses.append(py.Vector2(cur.cx, cur.cy))
        self.cursor_poses.append(new_cur_poses)

        new_code_poses = []
        for code in app.blockChars:
            new_code_poses.append(py.Vector2(code.cx, code.cy))
        self.code_block_poses.append(new_code_poses)
    
    def play(self, frame_rate):
        py.init()

        screen = py.display.set_mode((1280, 720))
        clock = py.time.Clock()
        running = True
        frame = 0

        while running and frame < len(self.player_pos):

            screen.fill("white")
            py.draw.circle(screen, "blue", self.player_pos[frame], 10)
            for pos in self.cursor_poses[frame]:
                py.draw.circle(screen, "black", pos, 5)
            for pos in self.code_block_poses[frame]:
                py.draw.circle(screen, "red", pos, 5)

            py.display.flip()
            frame += 1
            for event in py.event.get():
                if event.type == py.QUIT:
                    running = False

            clock.tick(frame_rate)
        
        py.quit()

class playerSprite:
    def __init__(self, design, cx, cy):
        self.body = design # 2D list of ASCII indexes
        self.walkingBody = copy.copy(self.body)
        self.cx = cx 
        self.cy = cy
        ##
        self.last_x = cx
        self.last_y = cy
        ##
        self.dx = 1

        self.moveSpeed = 6
        self.isWalking = False
        self.steppingOut = True
        self.hasFeet = True
        
        self.dashLength = 10
        self.dashCoolDown = 10
        self.dashAvailable = True
        self.dashEchoes = []

        self.smokeSpirals = []
        self.hasBuff = False
        self.buffs = dict() # mapping of buff name : time left

        self.isAttacking = False
        self.attackAngle = 0

        self.isDead = False

    def damage(self, row, col, amt, app):
        app.fitness -= 50
        #apply damage
        self.body[row][col] = max(self.body[row][col] - amt, 32) #cutoff at 32
        #check if row is empty
        if rowIsAll32s(self.body[row]):
            if self.hasFeet:
                rows = len(self.body)
                footRow = rows - 1
                if row == footRow: self.hasFeet = False
            self.body.pop(row)
        # check if col is empty
        elif colIsAll32s(self.body, col):
            removeCol(self.body, col)

    def flip(self):
        self.body = flippedHorizontally(self.body)

    def move(self, app, dy, dx): 
        self.cx += self.moveSpeed * dx
        self.cy += self.moveSpeed * dy
        if not spriteIsLegal(app):
            self.cx -= self.moveSpeed * dx
            self.cy -= self.moveSpeed * dy
            return False
        return True
    
    def walk(self, app):
        self.walkingBody = copy.deepcopy(self.body) # dont mess with og matrix
        left = True if self.dx == 1 else False # more readable direction
        rows = len(self.body)
        footRow = rows - 1
        # how to animate if one foot left
        if self.hasFeet and feetCount(self.body[footRow]) == 1:
            footIndex = self.body[footRow].index(124)
            if app.player.steppingOut:
                self.walkingBody[footRow][footIndex] = 47 if left else 92 #/ or \
            else:
                self.walkingBody[footRow][footIndex] = 124 # |
        #how to animate if two feet left 
        elif self.hasFeet and feetCount(self.body[footRow]) == 2:
            footIndex1 = getFirstFootIndex(self.body[footRow])
            footIndex2 = footIndex1 + 1
            steppingFoot = footIndex1 if app.player.steppingOut else footIndex2
            standingFoot = footIndex2 if app.player.steppingOut else footIndex1
            self.walkingBody[footRow][standingFoot] = 124 # |
            self.walkingBody[footRow][steppingFoot] = 47 if left else 92 #/ or \

    def dash(self, app, dy, dx):
        i = 0
        while self.move(app, dy, dx):
            i += 1
            if i == self.dashLength:
                break
    
    #buffs
    def spawnSmoke(self, app):
        left, top = getPlayerLeftTop(app)
        rows, cols = len(self.body), len(self.body[0])
        playerWidth, playerHeight = cols * app.cellSize, rows * app.cellSize
        right, bottom = left+ playerWidth, top + playerHeight
        cy = bottom
        cx = random.randrange(int(left), int(right))
    
    def addBuff(self, name, app):
        #applying buff effects:
        if name not in self.buffs:
            if name == 'green':
                self.dashLength *= 2
            if name == 'blue':
                self.moveSpeed *= 1.5
            if name == 'purple':
                self.dashCoolDown /= 2
        #adding buff to time dictionary
        self.buffs[name] = app.buffDuration
        self.hasBuff = True

    
    def revertBuff(self, buff):
        #reversing effects of given buff
        if buff == 'green':
            self.dashLength /= 2
        if buff == 'blue':
            self.moveSpeed /=1.5
        if buff == 'purple':
            self.dashCoolDown *= 2
        
    
    def decayBuffs(self):
        if len(self.buffs) == 0:
            self.hasBuff = False
        for buff in self.buffs:
            if self.buffs[buff] == 0:
                self.revertBuff(buff)
                self.buffs.pop(buff)
                break
            else:
                self.buffs[buff] -= 1
    
    ## attack
    def attack(self, mouseX, mouseY):
        self.attackAngle = getAngle(self.cx, self.cy, mouseX, mouseY)
        self.isAttacking = True

    def animateAttack(self, app):
        sweepConeWidth = app.sweepConeWidth
        if app.attackCounter == 0:
            self.startAngle = self.attackAngle-(sweepConeWidth/2) + 10
            self.sweepAngle = sweepConeWidth
        elif 4 > app.attackCounter >= 3:
            self.startAngle = self.attackAngle-(sweepConeWidth/2)
            self.sweepAngle = sweepConeWidth/1.9
        elif 6 > app.attackCounter >= 4:
            self.startAngle = self.attackAngle-(sweepConeWidth/2)
            self.sweepAngle = sweepConeWidth/2.7
        app.attackCounter += 1
        if app.attackCounter >= 6:
            self.isAttacking = False
            app.attackCounter = 0

def normalize(vect):
    norm = np.linalg.norm(vect)
    if norm == 0: 
       return vect
    return vect / norm

class cursor:
    def __init__(self, app, cx, cy, dx, dy, accelerationx, accelerationy):
        self.cx = cx
        self.cy = cy
        self.dx = dx
        self.dy = dy
        self.accelerationx = accelerationx
        self.accelerationy = accelerationy
        self.angle = getAngle(self.cx, self.cy, app.player.cx, app.player.cy)
        self.lifeSpan = 120
        self.vel = np.array([0,0])

    def move(self):
        self.cx += self.dx
        self.cy += self.dy
        self.dx += self.accelerationx
        self.dy += self.accelerationy
    
    def updateAngle(self, app):
        self.angle = getAngle(self.cx, self.cy, app.player.cx, app.player.cy)

class arcCursor(cursor):
    def __init__(self, app, cx, cy, dx, dy, accelerationx, 
                 accelerationy, speed = 0.13, amplitude = 7):
        super().__init__(app, cx, cy, dx, dy, accelerationx, accelerationy)
        self.linearCx = cx
        self.linearCy = cy
        self.counter = 0
        self.speed = speed
        self.amplitude = amplitude

    
    def move(self):
        self.linearCx += self.dx
        self.linearCy += self.dy
        self.cx = self.linearCx + self.dy*math.sin(self.counter)*self.amplitude
        self.cy = self.linearCy + self.dx*math.sin(self.counter)*self.amplitude
        self.dx += self.accelerationx
        self.dy += self.accelerationy
        self.counter += self.speed
        self.amplitude += self.speed*3

class bouncyCursor(cursor):
    def __init__(self, app, cx, cy, dx, dy, accelerationx, accelerationy):
        super().__init__(app, cx, cy, dx, dy, accelerationx, accelerationy)
        self.bottomBound = app.terminalTop + app.terminalHeight

    
    def move(self):
        self.cx += self.dx
        self.cy += self.dy
        self.dx += self.accelerationx
        self.dy += self.accelerationy
        if self.cy > self.bottomBound:
            self.dy *= -1 #flip direction

class zigZagCursor(cursor): 
    def __init__(self, app, cx, cy, dx, dy):
        self.lifeSpan = 120
        self.cx = cx
        self.cy = cy
        self.dx = dx
        self.dy = dy
        self.angle = getAngle(self.cx, self.cy, app.player.cx, app.player.cy)
        self.topBound = app.terminalTop
        self.bottomBound = app.terminalTop + app.terminalHeight
    
    def move(self):
        self.cx += self.dx
        self.cy += self.dy
        if self.cy <= self.topBound or self.cy >= self.bottomBound:
            self.dy *= -1


class powerUp:
    def __init__(self, cx, cy, type):
        self.cx = cx
        self.cy = cy
        self.type = type

class codeBlock:

    def __init__(self, app, cx, cy, string):
        self.left = cx - app.cellSize*(len(string)/2)
        self.cy = cy
        self.string = string
        self.typed = False
        self.i = 0
    
    def type(self, app): # type ith letter
        if self.i < len(self.string):
            codeCharLeft = self.left + self.i*app.cellSize
            spawnCodeChar(app, codeCharLeft, self.cy, self.string[self.i])
            self.i += 1
        else:
            self.typed = True



class blockChar:
    def __init__(self, cx, cy, c):
        self.cx = cx
        self.cy = cy
        self.c = c

class healthPack:
    def __init__(self, cx, cy, hp):
        self.cx = cx
        self.cy = cy
        self.hp = 10
    
    def healPlayer(self, app):
        body = (app.player.body)
        rows, cols = len(body), len(body[0])
        for row in range(rows):
            for col in range(cols):
                if body[row][col] != 32 and body[row][col] < 126:
                    app.player.body[row][col] += 1

####### CLASS-RELATED UTILITY FUNCTIONS

def getEllipseRadius(a, b, theta):
    return ((a*b)/((a**2) * (math.sin(theta)**2) + (b**2) 
                   * (math.cos(theta)**2))**0.5) # ellipse formula

def toRad(degrees):
    return degrees * (math.pi/180)
    
def toDegrees(rad):
    return rad * (180/math.pi)


def getAngle(cx, cy, mouseX, mouseY):
    clickdistance = (distance(cx, cy, mouseX, mouseY))
    if clickdistance == 0:
        return 0
    horizontal = mouseX - cx
    theta = math.acos(horizontal/clickdistance)
    theta *= 180/math.pi
    if mouseY > cy:
        theta = 180 - theta
        theta += 180
    return theta

##





#####


def feetCount(row):
    return row.count(124)

def getFirstFootIndex(row):
    cols = len(row)
    for col in range(cols):
        if row[col] != 32: return col
    return
    
 #####
 
def rowIsAll32s(row):
        cols = len(row)
        for col in range(cols):
            if row[col] != 32:
                return False
        return True
    
def colIsAll32s(L, col):
        rows = len(L)
        for row in range(rows):
            if L[row][col] != 32:
                return False
        return True

def removeCol(L, col):
    rows = len(L)
    for row in range(rows):
        L[row].pop(col)


################################# FLIPPING
def flippedHorizontally(L):
    rows = len(L)
    flippedBody = []
    for row in range(rows):
        # get row while changing direction of /, \, <, > symbols (odd)
        currentRow = [x if isNotOddSymbol(x) else oddSymbolSwap(x) 
                      for x in L[row]]
        #flip row
        flippedRow = list(reversed(currentRow))
        flippedBody += [flippedRow]
    return flippedBody

def isNotOddSymbol(x):
    return x != 47 and x != 92 and x != 60 and x!= 62

def oddSymbolSwap(x):
    if x == 47: return 92
    if x == 92: return 47
    if x == 60: return 62
    if x == 62: return 60

######################## DRAWING

def getCellLeftTop(app, row, col):
    playerLeft, playerTop = getPlayerLeftTop(app)
    cellLeft = playerLeft + (col * app.cellSize)
    cellTop = playerTop + (row * app.cellSize)
    return (cellLeft, cellTop)

def getPlayerLeftTop(app):
    if len(app.player.body) <= 0: return 0, 0
    rows, cols = len(app.player.body), len(app.player.body[0])
    playerWidth, playerHeight = cols * app.cellSize, rows * app.cellSize
    left = app.player.cx - playerWidth//2
    top =  app.player.cy - playerHeight//2
    return left, top

####################### KEY PRESS
def onKeyPress(app, key):
    if not app.player.isDead:
        if key == 'a':
            if app.player.dx != -1:
                app.player.flip()
            app.player.dx = -1
        elif key == 'd':
            if app.player.dx != 1:
                app.player.flip()
            app.player.dx = 1

        if app.player.dashAvailable:
            if key == 'space' and not app.verticalPressed:
                app.player.dashAvailable = False
                app.player.dash(app, 0, app.player.dx)
            elif key == 'space' and not app.sidePressed:
                app.player.dashAvailable = False
                app.player.dash(app, app.player.dy, 0)
            elif key == 'space' and app.verticalPressed and app.sidePressed:
                app.player.dashAvailable = False
                app.player.dash(app, app.player.dy, app.player.dx)
        
        #DEVTEST - UNCOMMENT TO PLAY W/ SPAWNING
        '''
        if key == 'g':
            app.player.addBuff('green', app)
        if key == 'b':
            app.player.addBuff('blue', app)
        if key == 'm':
            app.player.addBuff('purple', app)

        if key == 'k':
            spawnPolygonCursors(app, app.player.cx, app.player.cy, 8, 100, 
            0.10, isUniform = True, type = 'cursor')
        if key == 'l':
            spawnLinearCursors(app, app.player.cx, app.player.cy, 6, 200, 500, 
            0.50, 80, isUniform = False, type = 'cursor')

        if key == 'j':
            spawnCodeBlock(app, 'meow')
        
        if key == 'z':
            spawnZigZagCursor(app, 30, 3)
        
        if key == 'i':
            spawnPowerUp(app, 'green')
        if key == 'y':
            app.storyStep = 9300 # choose step from script
        '''
        
def onKeyRelease(app, keys):
    app.player.isWalking = False
    app.sidePressed = False
    app.verticalPressed = False

def onKeyHold(app, keys):
    if not app.player.isDead:
        if 'a' or 'd' or 's' or 'w' in keys:
            app.player.isWalking = True
        if ('a' not in keys and 'd' not in keys and 's' not in keys 
            and 'w' not in keys):
            app.player.isWalking = False
        if 'a' in keys:
            app.sidePressed = True
            app.player.move(app, 0, -1)
        if 'd' in keys:
            app.sidePressed = True
            app.player.move(app, 0, 1)
        if 's' in keys:
            app.verticalPressed = True
            app.player.dy = 1
            app.player.move(app, 1, 0)
        if 'w' in keys:
            app.verticalPressed = True
            app.player.dy = -1
            app.player.move(app, -1, 0)


def spriteIsLegal(app):
    if len(app.player.body) == 0:
        return True
    rows, cols = len(app.player.body), len(app.player.body[0])
    left, top = getPlayerLeftTop(app)
    right = left + cols * app.cellSize
    bottom = top + rows * app.cellSize
    return (left > app.playerBoundLeft and right < app.playerBoundRight 
            and bottom < app.playerBoundBottom and top > app.playerBoundTop)

############################ MOUSE PRESS


def onMousePress(app, mouseX, mouseY):
    if not app.player.isDead:
        if isInPlayerHitbox(app, mouseX, mouseY):
            row, col = getPlayerCellTouched(app, mouseX, mouseY)
            if row != None and col != None:
                app.player.damage(row, col, app.personalMouseDMG, app)
        else:
            app.player.attack(mouseX, mouseY)

def distance(x0, y0, x1, y1):
    return ((y1 - y0)**2 + (x1 - x0)**2)**0.5

def isInPlayerHitbox(app, x, y): #MOVE TO COLLISION DETECTION
    cellWidth = cellHeight = app.cellSize
    leftCoord, topCoord = getPlayerLeftTop(app)
    playerRows, playerCols = len(app.player.body), len(app.player.body[0])
    rightCoord = leftCoord + playerCols * cellWidth
    bottomCoord =  topCoord + playerRows * cellHeight
    return (leftCoord <= x <= rightCoord) and (topCoord <= y <= bottomCoord)

def getPlayerCellTouched(app, x, y):
    leftCoord, topCoord = getPlayerLeftTop(app)
    dx = x - leftCoord
    dy = y - topCoord
    cellWidth = cellHeight = app.cellSize
    row = math.floor(dy / cellHeight)
    col = math.floor(dx / cellWidth)
    playerRows, playerCols = len(app.player.body), len(app.player.body[0])
    if (0 <= row < playerRows) and (0 <= col < playerCols):
      return (row, col)
    else:
      return (None, None)

def getClosestCodeBlock(app):
    def key(pos):
        return np.linalg.norm(np.array([app.player.cx, app.player.cy]) - np.array([pos.cx, pos.cy]))
    codes = sorted(app.blockChars, key=key)
    if len(codes) == 0:
        return -1, -1
    return codes[0].cx, codes[0].cy

############################### ON STEP

def onStep(app):
    app.replay.capture(app)
    if len(app.player.body) == 0:
        # game over
        app.player.isDead = True
    if not app.player.isDead:

        atx, aty = getClosestCodeBlock(app)
        atxy = np.array([atx, aty])
        playerxy = np.array([app.player.cx, app.player.cy])
        mouse_pos = normalize(atxy - playerxy) * 50 + playerxy
        onMousePress(app, mouse_pos[0], mouse_pos[1])

        if app.player.isAttacking:
            app.player.animateAttack(app)
            checkPlayerAttackCollision(app)
            removeCodeBlocks(app)
        if app.player.isWalking and app.player.hasFeet:
            animateWalk(app)
        if len(app.player.dashEchoes) > 0:
            fadeEchoes(app)
        if len(app.powerUps) > 0:
            checkPowerUpCollision(app)
        if len(app.healthPacks) > 0:
            checkHealthPackCollision(app)
        if app.player.hasBuff:
            spawnSmokeSpirals(app)
            app.player.decayBuffs()    
        if len(app.player.smokeSpirals) > 0:
            animateSmoke(app)
        if not app.player.dashAvailable:
            resetDash(app)
        if len(app.cursors) > 0:
            moveCursors(app)
            checkCursorCollision(app)
        if len(app.codeBlocks) != 0:
            for block in app.codeBlocks:
                if not block.typed:
                    block.type(app)
        elif app.gameMode == 'infinite': #spawn patterns in infinite
            spawnCodeBlock(app, random.choice(loadDesign('codeBlocks')))
        if app.gameMode == 'infinite':
            manageCodeSpawning(app)
            manageCursorSpawning(app)
            managePowerUpsSpawning(app)
        ###


##############STORY MODE - unforunately 'elif' chain only way to run story

#########INFINITE MODE

def managePowerUpsSpawning(app):
    threshhold = 1000
    if app.powerUpSpawnCounter > threshhold:
        color = random.choice(['blue', 'green', 'purple'])
        spawnPowerUp(app, color)
        app.powerUpSpawnCounter = 0
    else:
        app.powerUpSpawnCounter += 1


def manageCursorSpawning(app):
    if app.cursorSpawnCounter > app.cursorSpawnThreshhold:
        for codeblock in app.codeBlocks:
            spawnNewAttack(app, codeblock.string)
        app.cursorSpawnCounter = 0
        if app.cursorSpawnThreshhold > 40:
            app.cursorSpawnThreshhold -= 1 # MAKE THE GAME FASTER!
    else:
        app.cursorSpawnCounter += 1

def removeCodeBlocks(app):
    letterSet = set()
    for char in app.blockChars:
        letterSet.add(char.c)
    i = 0
    while i < len(app.codeBlocks):
        if not app.codeBlocks[i].typed:
            i += 1
            continue
        codeBlockBroken = False
        for letter in app.codeBlocks[i].string:
            if letter not in letterSet:
                codeBlockBroken = True
        if codeBlockBroken:
            app.codeBlocks.pop(i)
            app.points += 1
            app.fitness += 1000
        else:
            i += 1



def spawnNewAttack(app, string):
        if string == 'pip':
            spawnPipAttack(app)
        elif string == 'cd':
            spawnCdAttack(app)
        elif string == 'EXIT':
            spawnExitAttack(app)
        else: #codeblock is break
            spawnBreakAttack(app)


def spawnPipAttack(app):
    i = random.randint(1,5)
    n = random.randint(3, 8)
    r = random.randint(100, 600)
    if i == 0:
        spawnPolygonCursors(app, app.player.cx, app.player.cy, n, r, 0.50, 
                            type = 'arcCursor')
    else:
        spawnPolygonCursors(app, app.player.cx, app.player.cy, n, r, 0.50, 
                            type = 'cursor')

def spawnCdAttack(app):
    i = random.randint(1,5)
    n = random.randint(3, 8)
    r = random.randint(100, 600)
    l = random.randint(100, 700)
    a = random.randint(1, 360)
    if i == 0:
        for j in range(n):
            spawnZigZagCursor(app, app.player.cx, 2+j)
    else:
        spawnLinearCursors(app, app.player.cx, app.player.cy, n, r, l, 0.50, a, 
                           isUniform = True, type = 'cursor')

def spawnBreakAttack(app):
    n = random.randint(3, 8)
    r = random.randint(100, 600)
    l = random.randint(100, 700)
    spawnLinearCursors(app, app.player.cx, app.player.cy, n, r, l, 0.50, 80, 
                       type = 'bouncyCursor')

def spawnExitAttack(app):
    n = random.randint(4, 8)
    r = random.randint(100, 600)
    for i in range(2, n):
        spawnPolygonCursors(app, app.player.cx, app.player.cy, i, r, 0.50, 
                            type = 'arcCursor')



def manageCodeSpawning(app):
    codeBlockSpawnThreshhold= 1000
    if app.codeBlockSpawnCounter > codeBlockSpawnThreshhold:
        spawnCodeBlock(app, random.choice(loadDesign('codeBlocks')))
        app.codeBlockSpawnCounter = 0
    else:
        app.codeBlockSpawnCounter += 1


def getRandomCoords(app):
    cx = (random.randint(round(app.playerBoundLeft + 10), 
                         round(app.playerBoundRight -10)))
    cy = (random.randint(round(app.playerBoundTop + 10), 
                         round(app.playerBoundBottom - 10)))
    return cx, cy

def spawnPowerUp(app, type = 'green'):
    cx, cy = getRandomCoords(app)
    app.powerUps+= [powerUp( cx, cy, type)]


def spawnHealthPack(app):
    cx, cy = getRandomCoords(app)
    app.healthPacks+= [healthPack( cx, cy, type)]


def checkHealthPackCollision(app):
    i = 0
    while i < len(app.healthPacks):
        pack = app.healthPacks[i]
        if isInPlayerHitbox(app, pack.cx, pack.cy):
            pack.healPlayer(app)
            pack.hp -= 1
        if pack.hp == 0:
            app.healthPacks.pop(i)
        else:
            i += 1


def checkPowerUpCollision(app):
    i = 0
    while i < len(app.powerUps):
        powerUp = app.powerUps[i]
        if isInPlayerHitbox(app, powerUp.cx, powerUp.cy):
            app.player.addBuff(powerUp.type, app)
            app.powerUps.pop(i)
        else:
            i += 1


def checkPlayerAttackCollision(app):
    rows, cols = len(app.player.body), len(app.player.body[0])
    playerWidth, playerHeight = cols * app.cellSize, rows * app.cellSize
    i = 0
    while i < len(app.blockChars):
        char = app.blockChars[i]
        angle = getAngle(app.player.cx, app.player.cy, char.cx, char.cy)
        attackRadius = getEllipseRadius(playerWidth/2, playerHeight/2, angle)
        distanceFromPlayer = distance(char.cx, char.cy, 
                                      app.player.cx, app.player.cy)
        lesserBoundaryAngle = (app.player.attackAngle - 
                                        (app.sweepConeWidth+20))
        greaterBoundaryAngle = (app.player.attackAngle + 
                                        (app.sweepConeWidth +20))
        if (isInCounterClockWiseOrder(lesserBoundaryAngle, 
                                      angle, greaterBoundaryAngle) and 
                distanceFromPlayer < attackRadius + app.attackOffset):
            app.blockChars.pop(i)
        else:
            i += 1

def isInCounterClockWiseOrder(first, middle, last):
     first %= 360
     middle %= 360
     last %= 360
     if first <= last:
        return first <= middle <= last
     return(first<=middle) or (middle<=last)


def resetDash(app):
    if app.dashCounter >= app.player.dashCoolDown:
        app.player.dashAvailable = True
        app.dashCounter = 0
    else:
        app.dashCounter += 1
        
def spawnSmokeSpirals(app):
    smokeSpawnThreshhold = 20
    if app.smokeSpawnCounter >= smokeSpawnThreshhold:
        app.player.spawnSmoke(app)
        app.smokeSpawnCounter = 0
    else:
        app.smokeSpawnCounter += 1


def fadeEchoes(app):
    echoThreshhold = 3
    if app.echoFadeCounter >= echoThreshhold:
        adjustEchoTimes(app)
        app.echoFadeCounter = 0
    else:
        app.echoFadeCounter += 1

def adjustEchoTimes(app):
    i = 0
    while i < (len(app.player.dashEchoes)):
        currentEcho = app.player.dashEchoes[i]
        currentEcho.time += 1
        #remove old echoes
        if currentEcho.time == app.echoTimeLength:
            app.player.dashEchoes.pop(i)
        i += 1


def animateWalk(app):
    walkThreshhold = 3
    #getting (readable) direction
    ## pacing of footsteps
    if app.walkCounter >= walkThreshhold:
        app.player.steppingOut = not app.player.steppingOut
        app.walkCounter = 0
    else: 
        app.walkCounter += 1
    app.player.walk(app)

def animateSmoke(app):
    smokeThreshhold = 3
    if app.smokeCounter >= smokeThreshhold:
        moveSmoke(app)
        app.smokeCounter = 0
    else:
        app.smokeCounter += 1

def moveSmoke(app): #on step
    distanceThreshhold = 200
    i = 0
    while i < (len(app.player.smokeSpirals)):
        currentSmoke = app.player.smokeSpirals[i]
        if (currentSmoke.cy < currentSmoke.cyBound or 
            distance(currentSmoke.cx, currentSmoke.cy, 
                     app.player.cx, app.player.cy) 
            > distanceThreshhold):
            app.player.smokeSpirals.pop(i)
        else:
            if (currentSmoke.cx < currentSmoke.bound1 or 
                    currentSmoke.cx > currentSmoke.bound2):
                currentSmoke.changeDirection()
            currentSmoke.move()
            i += 1

### CURSOR ATTACKS

def checkCursorCollision(app):
    for currentCursor in app.cursors:
        if len(app.player.body) == 0:
            app.player.isDead = True
            break
        x, y = currentCursor.cx, currentCursor.cy
        if isInPlayerHitbox(app, x, y):
            row, col = getPlayerCellTouched(app, x, y)
            if row != None and col != None:
                app.player.damage(row, col, app.personalMouseDMG, app)


def moveCursors(app):
    i = 0
    while i < (len(app.cursors)):
        currentCursor = app.cursors[i]
        currentCursor.updateAngle(app)
        currentCursor.lifeSpan -= 1
        if (currentCursor.lifeSpan <= 0):
            app.cursors.pop(i)
        else:
            currentCursor.move()
        i += 1

def spawnPolygonCursors(app, cx, cy, N, r, acceleration, isUniform=True, 
                        type = 'cursor'):
    bounceFactor = 4
    for n in range(N): # uses roots of unity to calculate pos
        cursorx, cursory = (cx + r*math.cos(toRad((360/N)*n)), 
                                cy + r*math.sin(toRad((360/N)*n)))
        xDistanceFromCx, yDistanceFromCy = (cx - cursorx), (cy - cursory)
        dx, dy = xDistanceFromCx/r, yDistanceFromCy/r
        if isUniform:
            accelerationX = dx * acceleration
            accelerationY = dy * acceleration
        else:
            accelerationX = dx * acceleration + dx * 1/(n+0.1)
            accelerationY = dy * acceleration + dx * 1/(n+0.1)
        if type == 'cursor':
            app.cursors += [cursor(app, cursorx, cursory, bounceFactor*-dx, 
                                   bounceFactor*-dy, accelerationX, 
                                   accelerationY)]
        elif type == 'arcCursor':
            app.cursors += [arcCursor(app, cursorx, cursory, 
                                      dx, dy, accelerationX, accelerationY)]
    
def spawnLinearCursors(app, cx, cy, amt, r, lineLength, acceleration, angle=180, 
                            isUniform=True, type = 'cursor'):
    bounceFactor = 4
    seperation = lineLength / amt-1
    xToMidPoint, yToMidPoint = r*math.cos(toRad(angle)), r*math.sin(toRad(angle))
    midCursorX, midCursorY = (cx + xToMidPoint), (cy - yToMidPoint)
    run, rise = yToMidPoint/r, xToMidPoint/r #perpendicular slope = -reciprocal
    dx, dy = -1*rise, run
    startPointx, startPointy = (midCursorX - ((amt-1)/2)*run*seperation, 
                                midCursorY - ((amt-1)/2)*rise*seperation)
    for i in range(amt):
        cursorx, cursory = (startPointx + i*run*seperation, 
                            startPointy + i*rise*seperation)
        if isUniform:
            accelerationX = dx * acceleration
            accelerationY = dy * acceleration
        else:
            accelerationX = dx * acceleration + dx * 1/(i+0.1)
            accelerationY = dy * acceleration + dx * 1/(i+0.1)
        if type == 'cursor':
            app.cursors += [cursor(app, cursorx, cursory, bounceFactor*-dx, 
                                   bounceFactor*-dy, accelerationX, 
                                    accelerationY)]
        elif type == 'arcCursor':
            app.cursors += [arcCursor(app, cursorx, cursory, dx, dy, 
                                      accelerationX, accelerationY)]
        elif type == 'bouncyCursor':
            app.cursors += [bouncyCursor(app, cursorx, cursory, dx, dy, 
                                         accelerationX, accelerationY)]
        

def spawnZigZagCursor(app, cx, N, speed = 7): #where N is number of peaks
    dx = ((abs(app.player.cx-cx))/(2.5*N))*0.01*speed
    dy = app.terminalHeight*0.01 * speed
    app.cursors += [zigZagCursor(app, cx, app.player.cy, dx, dy)]

### SPAWNING CODE BLOCKS
def spawnCodeBlock(app, string):
    cx = random.randint(round(app.playerBoundLeft + 10), 
                        round(app.playerBoundRight - 10))
    cy = random.randint(round(app.playerBoundTop + 10), 
                        round(app.playerBoundBottom - 10))
    app.codeBlocks += [codeBlock(app, cx, cy, string)]


def spawnCodeChar(app, left, cy, c):
    cx = left + app.cellSize/2
    app.blockChars += [blockChar(cx, cy, c)]

######################## DESIGNS

def loadDesign(name):
    # name -> design
    catalog = { 'player': [[32, 32, 95, 95, 32], 
                        [32, 95, 61, 61, 95], 
                        [32, 32, 124, 46, 62], 
                        [32, 32, 47, 32, 96], 
                        [32, 60, 42, 59, 92], 
                        [95, 45, 42, 95, 46], 
                        [32, 32, 124, 124, 32]],
                'codeBlocks': ['cd', 'pip', 'break', 'EXIT']
                        }
    return catalog[name]

class App:
    def __init__(self):
        self.fitness = 0
        self.replay = Replay()

        self.gameMode = 'infinite'

        self.width = 1024
        self.height = 768

        self.terminalLeft = 30
        self.terminalTop = self.height/15
        self.terminalWidth = 800
        self.terminalHeight = 650

        self.terminalcx, self.terminalcy = self.terminalLeft + self.terminalWidth/2, self.terminalTop + self.terminalHeight/2

        self.playerBoundLeft = self.terminalLeft + self.terminalWidth/15
        self.playerBoundTop = self.terminalTop + self.terminalHeight/15
        self.playerBoundRight = (self.terminalLeft + self.terminalWidth 
                                - self.terminalWidth/15)
        self.playerBoundBottom = (self.terminalTop + self.terminalHeight 
                                - self.terminalHeight/15)

        self.cellSize = 11 #px
        self.playerDesign = loadDesign('player')
        self.player = playerSprite(self.playerDesign, self.terminalcx, self.terminalcy) 
        
        self.walkCounter = 0
        self.echoFadeCounter = 0
        self.smokeCounter = 0
        self.smokeSpawnCounter = 0
        self.dashCounter = 0
        self.attackCounter = 0
        self.blockCharCounter = 0
        self.cursorSpawnCounter = 0
        self.cursorSpawnThreshhold = 90
        self.codeBlockSpawnCounter = 0
        self.powerUpSpawnCounter = 0
        
        self.points = 0
        
        self.cursors = []

        self.codeBlocks = []
        self.isTypingCodeBlock = False
        self.blockChars = []
        self.powerUps = []
        self.healthPacks = []

        self.sweepConeWidth = 70
        self.attackOffset = 40

        self.echoTimeLength = 60
        self.personalMouseDMG = 10
        self.sidePressed = False
        self.verticalPressed = False

        self.buffDuration = 1000

def getFirstIndex(arr, target):
    for i in range(len(arr)):
        if arr[i] == target: return i
    return -1

def angleBetweenVectors(A, B):
    # Convert the vectors to NumPy arrays
    A = np.array(A)
    B = np.array(B)

    # Calculate the dot product of A and B
    dot_product = np.dot(A, B)

    # Calculate the magnitudes of A and B
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    if magnitude_A == 0 or magnitude_B == 0: return 0

    # Calculate the cosine of the angle between A and B
    cosine_theta = dot_product / (magnitude_A * magnitude_B)

    # Use arccosine to find the angle in radians
    angle_radians = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

    return angle_radians

# up, down, right, left, dash
def handleInput(app, inpt):
    #code_x, code_y = getClosestCodeBlock(app)
    #onMousePress(app, code_x, code_y)
    # vertical = inpt[0:2]
    # highest_vertical = max(vertical)
    # vertical_ind = getFirstIndex(vertical, highest_vertical)
    # if highest_vertical > 0:
    #     if vertical_ind == 0:
    #         onKeyHold(app, 'w')
    #     else:
    #         onKeyHold(app, 's')

    # horizontal = inpt[2:4]
    # highest_horizontal = max(horizontal)
    # horizontal_ind = getFirstIndex(horizontal, highest_horizontal)
    # if highest_horizontal > 0:
    #     if horizontal_ind == 0:
    #         onKeyHold(app, 'd')
    #     else:
    #         onKeyHold(app, 'a')

    if inpt[0] > 0:
        onKeyHold(app, 'w')
    if inpt[1] > 0:
        onKeyHold(app, 's')
    if inpt[2] > 0:
        onKeyHold(app, 'd')
    if inpt[3] > 0:
        onKeyHold(app, 'a')
    if inpt[4] > 0:
        onKeyPress(app, 'space')


# cur_x, cur_y, vel_x, vel_y, dot, exists .... code_x, code_y, exists
def getOutput(app, cur_num):
    o = []
    def key(pos):
        return np.linalg.norm(np.array([app.player.cx, app.player.cy]) - np.array([pos.cx, pos.cy]))
    cursors = sorted(app.cursors, key=key)
    for i in range(cur_num):
        if i < len(cursors):
            #o.append((cursors[i].cx - app.player.cx)/1000)
            #o.append((cursors[i].cy - app.player.cy)/1000)
            cursor_dist = (1000 - distance(app.player.cx, app.player.cy, cursors[i].cx, cursors[i].cy) * 2)/1000
            cursor_angle = math.atan2(cursors[i].cy - app.player.cy, cursors[i].cx - app.player.cx)/math.pi
            o.append(cursor_dist)
            o.append(cursor_angle)
            vel = np.array([cursors[i].dx, cursors[i].dy])
            to_player = np.array([app.player.cx - cursors[i].cx, app.player.cy - cursors[i].cy])
            vel_angle = angleBetweenVectors(vel, to_player)/math.pi
            o.append(vel_angle)
            #o.append(vel[0])
            #o.append(vel[1])
            #dir_vect = normalize(np.array([app.player.cx, app.player.cy]) - np.array([cursors[i].cx, cursors[i].cy]))
            #o.append(np.dot(dir_vect, vel))
            #o.append(1)
        else:
            o.extend([-2,-2,-1])
    
    codes = sorted(app.blockChars, key=key)

    if len(codes) == 0:
        o.extend([0])
    else:
        # o.append((codes[0].cx - app.player.cx)/1000)
        # o.append((codes[0].cy - app.player.cy)/1000)
        # code_dist = (1000 - distance(app.player.cx, app.player.cy, codes[0].cx, codes[0].cy))/1000

        # if distance(app.player.cx, app.player.cy, codes[0].cx, codes[0].cy) <= \
        # distance(app.player.last_x, app.player.last_y, codes[0].cx, codes[0].cy):
        #     app.fitness -= 50

        code_angle = math.atan2(codes[0].cy - app.player.cy, codes[0].cx - app.player.cx)/math.pi
        #o.append(code_dist)
        o.append(code_angle)
        #o.append(1)
    
    mid_x_dist, mid_y_dist = (app.player.cx - app.terminalcx)/318, (app.player.cy - app.terminalcy)/240
    o.append(mid_x_dist)
    o.append(mid_y_dist)
    
    
    #print("out:")
    #print(o)
    
    return o

def getDistanceFitness(app):
    codex, codey = getClosestCodeBlock(app)
    codexy = np.array([codex, codey])
    playerxy = np.array([app.player.cx, app.player.cy])
    fit = max(0, 1000 - np.linalg.norm(codexy - playerxy))
    app.fitness += fit

def main():
    py.init()

    screen = py.display.set_mode((1280, 720))
    clock = py.time.Clock()
    running = True

    app = App()
    while running:
        
        onStep(app)
        onKeyHold(app, "a")
        onKeyHold(app, "s")
        print(getOutput(app, 4))
        #getOutput(app, 4)
        if app.player.isDead:
            app = App()

        screen.fill("white")
        py.draw.circle(screen, "red", py.Vector2(app.player.cx, app.player.cy), 10)
        for code in app.blockChars:
            py.draw.circle(screen, "black", py.Vector2(code.cx, code.cy), 5)
        for cur in app.cursors:
            py.draw.circle(screen, "black", py.Vector2(cur.cx, cur.cy), 5)

        py.display.flip()

        for event in py.event.get():
            if event.type == py.QUIT:
                running = False

        clock.tick(60)
        
if __name__ == "__main__":
    main()