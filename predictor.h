#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

#define OPPONENT_X_SPACE 100
#define OPPONENT_Y_SPACE 100
#define PLAYER_X_SPACE 100
#define PLAYER_Y_SPACE 100
#define BALL_X_SPACE 20
#define BALL_Y_SPACE 20
#define ANGLE_SPACE 10

#define OPPONENT_X_NUM 5
#define OPPONENT_Y_NUM 5
#define PLAYER_X_NUM 5
#define PLAYER_Y_NUM 5
#define BALL_X_NUM 10
#define BALL_Y_NUM 10
#define ANGLE_NUM (180/ANGLE_SPACE)

class objectInfo{
public:
	objectInfo(double x, double y);
	~objectInfo();
	double x,y;
};

class probInfo{
public:
	int label;
	double prob;
	static bool sortProbInfo(const probInfo& p1, const probInfo& p2);
};

class predictor{
public:
	predictor();
	~predictor();

	// feedInfo when ball is close to player under certain theshold T, and the return play angle
	void feedInformation(const objectInfo& ballInfo, const objectInfo& opponentInfo, const objectInfo& playerInfo, const double& returnAngle);
	void predictNthBestDirections(const objectInfo& ballInfo, const objectInfo& opponentInfo, const objectInfo& playerInfo, const int& bestNum, double& prob, int& angle);

	void showLearningStatus();

private:

	double opponentPos[OPPONENT_X_NUM][OPPONENT_Y_NUM][ANGLE_NUM];
	double playerPos[PLAYER_X_NUM][PLAYER_Y_NUM][ANGLE_NUM];
	double ballPos[BALL_X_NUM][BALL_Y_NUM][ANGLE_NUM];  // opponent x,y, ball x,y, player x,y
};