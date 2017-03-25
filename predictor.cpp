#include "predictor.h"

objectInfo::objectInfo(double x, double y){
	this->x = x;
	this->y = y;
};

objectInfo::~objectInfo(){
	;
}

bool probInfo::sortProbInfo(const probInfo& p1, const probInfo& p2){
	return p1.prob > p2.prob;
};

predictor::predictor(){
	
	for(int i = 0; i < OPPONENT_X_NUM; i++){
		for(int j = 0; j < OPPONENT_Y_NUM; j++){
			for(int k = 0; k < ANGLE_NUM; k++){
				opponentPos[i][j][k] = 0;
			}
		}
	}

	for(int i = 0; i < PLAYER_X_NUM; i++){
		for(int j = 0; j < PLAYER_Y_NUM; j++){
			for(int k = 0; k < ANGLE_NUM; k++){
				playerPos[i][j][k] = 0;
			}
		}
	}

	for(int i = 0; i < BALL_X_NUM; i++){
		for(int j = 0; j < BALL_Y_NUM; j++){
			for(int k = 0; k < ANGLE_NUM; k++){
				ballPos[i][j][k] = 0;
			}
		}
	}
}

predictor::~predictor(){
	;
}

void predictor::feedInformation(const objectInfo& ballInfo, const objectInfo& opponentInfo, const objectInfo& playerInfo, const double& returnAngle){
	
	int opponentX = (int)opponentInfo.x/OPPONENT_X_SPACE >= OPPONENT_X_NUM ? OPPONENT_X_NUM - 1 : (int)opponentInfo.x/OPPONENT_X_SPACE;
	int opponentY = (int)opponentInfo.y/OPPONENT_Y_SPACE >= OPPONENT_Y_NUM ? OPPONENT_Y_NUM - 1 : (int)opponentInfo.y/OPPONENT_X_SPACE;
	int playerX = (int)playerInfo.x/PLAYER_X_SPACE >= PLAYER_X_NUM ? PLAYER_X_NUM - 1 : (int)playerInfo.x/PLAYER_X_SPACE;
	int playerY = (int)playerInfo.y/PLAYER_Y_SPACE >= PLAYER_Y_NUM ? PLAYER_Y_NUM - 1 : (int)playerInfo.y/PLAYER_Y_SPACE;
	int ballX = (int)ballInfo.x/BALL_X_SPACE >= BALL_X_NUM ? BALL_X_NUM - 1 : (int)ballInfo.x/BALL_X_SPACE;
	int ballY = (int)ballInfo.y/BALL_Y_SPACE >= BALL_Y_NUM ? BALL_Y_NUM - 1 : (int)ballInfo.y/BALL_Y_SPACE;
	int angle = (int)returnAngle/ANGLE_SPACE >= ANGLE_NUM ? ANGLE_NUM - 1 : (int)returnAngle/ANGLE_NUM;

	opponentPos[opponentX][opponentY][angle] += 1;
	playerPos[playerX][playerY][angle] += 1;
	ballPos[ballX][ballY][angle] += 1;

	if(opponentX - 1 >= 0) opponentPos[opponentX-1][opponentY][angle] += 0.5;
	if(opponentX + 1 < OPPONENT_X_NUM) opponentPos[opponentX+1][opponentY][angle] += 0.5;
	if(opponentY - 1 >= 0) opponentPos[opponentX][opponentY-1][angle] += 0.5;
	if(opponentY + 1 < OPPONENT_Y_NUM) opponentPos[opponentX][opponentY+1][angle] += 0.5;


	if(playerX - 1 >= 0) playerPos[playerX-1][playerY][angle] += 0.5;
	if(playerX + 1 < PLAYER_X_NUM) playerPos[playerX+1][playerY][angle] += 0.5;
	if(playerY - 1 >= 0) playerPos[playerX][playerY-1][angle] += 0.5;
	if(playerY + 1 < PLAYER_Y_NUM) playerPos[playerX][playerY+1][angle] += 0.5;

	if(ballX - 1 >= 0) ballPos[ballX-1][ballY][angle] += 0.5;
	if(ballX + 1 < BALL_X_NUM) ballPos[ballX+1][ballY][angle] += 0.5;
	if(ballY - 1 >= 0) ballPos[ballX][ballY-1][angle] += 0.5;
	if(ballY + 1 < BALL_Y_NUM) ballPos[ballX][ballY+1][angle] += 0.5;

	if(angle - 1 >= 0){
		opponentPos[opponentX][opponentY][angle-1] += 0.5;
		playerPos[playerX][playerY][angle-1] += 0.5;
		ballPos[ballX][ballY][angle-1] += 0.5;
	}
	if(angle + 1 < ANGLE_NUM){
		opponentPos[opponentX][opponentY][angle+1] += 0.5;
		playerPos[playerX][playerY][angle+1] += 0.5;
		ballPos[ballX][ballY][angle+1] += 0.5;
	}

};

void predictor::predictNthBestDirections(const objectInfo& ballInfo, const objectInfo& opponentInfo, const objectInfo& playerInfo, const int& bestNum, double& prob, int& angle){

	int opponentX = (int)opponentInfo.x/OPPONENT_X_SPACE >= OPPONENT_X_NUM ? OPPONENT_X_NUM - 1 : (int)opponentInfo.x/OPPONENT_X_SPACE;
	int opponentY = (int)opponentInfo.y/OPPONENT_Y_SPACE >= OPPONENT_Y_NUM ? OPPONENT_Y_NUM - 1 : (int)opponentInfo.y/OPPONENT_X_SPACE;
	int playerX = (int)playerInfo.x/PLAYER_X_SPACE >= PLAYER_X_NUM ? PLAYER_X_NUM - 1 : (int)playerInfo.x/PLAYER_X_SPACE;
	int playerY = (int)playerInfo.y/PLAYER_Y_SPACE >= PLAYER_Y_NUM ? PLAYER_Y_NUM - 1 : (int)playerInfo.y/PLAYER_Y_SPACE;
	int ballX = (int)ballInfo.x/BALL_X_SPACE >= BALL_X_NUM ? BALL_X_NUM - 1 : (int)ballInfo.x/BALL_X_SPACE;
	int ballY = (int)ballInfo.y/BALL_Y_SPACE >= BALL_Y_NUM ? BALL_Y_NUM - 1 : (int)ballInfo.y/BALL_Y_SPACE;
	
	double total1 = 0.0;
	double total2 = 0.0;
	double total3 = 0.0;
	for(int i = 0; i < ANGLE_NUM; i++){
		total1 += opponentPos[opponentX][opponentY][i];
		total2 += playerPos[playerX][playerY][i];
		total3 += ballPos[ballX][ballY][i];
	}

	vector<probInfo> vProbs;
	for(int i = 0; i < ANGLE_NUM; i++){
		probInfo pbInfo;
		pbInfo.label = i;

		double prob1 = total1 == 0 ? 0 : opponentPos[opponentX][opponentY][i]/total1;
		double prob2 = total2 == 0 ? 0 : playerPos[playerX][playerY][i]/total2;
		double prob3 = total3 == 0 ? 0 : ballPos[ballX][ballY][i]/total3;

		pbInfo.prob = (prob1 + prob2 + prob3)/3.0;

    cerr << prob1 << "\t" << prob2 << "\t" << prob3 << "\t" << pbInfo.prob << endl;
		vProbs.push_back(pbInfo);
	}
	sort(vProbs.begin(), vProbs.end(), probInfo::sortProbInfo);

	prob = vProbs[bestNum].prob;
	angle = vProbs[bestNum].label*ANGLE_SPACE;
};

void predictor::showLearningStatus(){

	cerr << "OPPONENT MODEL STATUS: " << endl;

	for(int i = 0; i < OPPONENT_X_NUM; i++){
		for(int j = 0; j < OPPONENT_Y_NUM; j++){
			//cerr << "Opponent Pos [" << i << ":" << j << "]" << endl;
			for(int k = 0; k < ANGLE_NUM; k++){
				cerr << opponentPos[i][j][k] << " ";
			}
			cerr << endl;
		}
	}

	cerr << "PLAYER MODEL STATUS: " << endl;
	for(int i = 0; i < PLAYER_X_NUM; i++){
		for(int j = 0; j < PLAYER_Y_NUM; j++){
			//cerr << "Player Pos [" << i << ":" << j << "]" << endl;
			for(int k = 0; k < ANGLE_NUM; k++){
				cerr << playerPos[i][j][k] << " ";
			}
			cerr << endl;
		}
	}

	cerr << "BALL MODEL STATUS: " << endl;
	for(int i = 0; i < BALL_X_NUM; i++){
		for(int j = 0; j < BALL_Y_NUM; j++){
			//cerr << "Ball Pos [" << i << ":" << j << "]" << endl;
			for(int k = 0; k < ANGLE_NUM; k++){
				cerr << ballPos[i][j][k] << " ";
			}
			cerr << endl;
		}
	}
};
