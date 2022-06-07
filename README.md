Purpose of the Research
In this study, it is aimed to determine the potential winning strategies of chess game by considering game attributes such as duration of the game, openings, type of game (rated or regular) and chess sides (black and white). Determination process includes over 20.000 online match histories of players at different levels of chess.

Dataset: Chess Game Dataset
https://www.kaggle.com/datasets/datasnaek/chess

This dataset includes 20.000+ chess matches which are obtained from Lichess.org and it has 16 different columns as follows:
	Game ID as id
	Rated (T/F) as rated
	Start time as created_at
	End time as last_move_at
	Number of turns as turns
	End game status as victory_status
	Winner as winner
	Time increment as increment_code
	White player ID as white_id
	White player Rating as white_rating
	Black player ID as black_id
	Black player Rating as black_rating
	All moves in standard chess notation as moves
	Opening ECO code as opening_eco
	Opening name as opening_name
	Opening ply as opening_ply
