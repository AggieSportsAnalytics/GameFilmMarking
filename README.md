# Soccer Player Tracker ⚽️

### 🏁 Determine player, ball and line location while watching soccer games

This soccer project tracks players and the ball in soccer game film. It is able to differentiate between two teams and a referee. It allows teams to follow the play as the game is built up with clear confidence of where their team members are. 

## 🔑 Key Features
- Player Tracking: The project employs object detection and tracking algorithms to identify and track the positions of players on the field throughout the game. This information is crucial for making offside determinations.
- Team Color Segmentation: The system also analyzes the colors of the player jerseys to distinguish between teams. By detecting the dominant colors on the players' uniforms, the algorithm can categorize them into teams.
- Goalkeeper and Referee Exclusion: Goalkeepers and referees are easily recognized by their distinct attire. The system filters them out from the player detection results, ensuring that their positions do not interfere with the offside calculations.
- Real-Time Video Analysis: The system can process live video feeds from soccer matches, enabling real-time offside detection during gameplay. It can also be applied to pre-recorded matches for analysis and review.


## 💻  Technology
- OpenCV
- NumPy
- YoloV8
