# NFL Player Statistics Data Dictionary

## Purpose
This document provides a comprehensive description of each column in the NFL player statistics dataset. It serves as a reference for data analysts, machine learning engineers, and developers working on NFL prediction systems. The dataset aggregates player performance metrics from games, enabling feature engineering for predictive models.

## Key Structure
- **Columns**: Each entry lists a column name followed by its description.
- **Usage**: Use `player_id` for joining with other data sources like `load_players()`.
- **Dependencies**: Relies on NFL play-by-play data from APIs such as playstats; integrates with machine learning pipelines for fantasy points and EPA calculations.

## Columns

player_id	                                              	    Player's gsis_id. Use this to join to other sources, e.g. load_players().
player_name	                                              	    Abbreviated name of player as provided by playstats api
player_display_name	                                            Name of player as provided by `load_players()`
position	                                                    Position of player as listed by NFL
position_group	                                              	Position group of player as listed by NFL
headshot_url	                                              	Player's nfl.com headshot URL
season	                                                    	Official NFL season
week	                                  	                    Game week number
season_type	                                              	    `REG` for regular season, `POST` for postseason
team	                                              	        Abbreviation of player's team
opponent_team	                                              	Abbreviation of opponent's team
completions	                                  	                The number of completed passes.
attempts	                                  	                The number of pass attempts as defined by the NFL.
passing_yards	                                  	            Yards gained on pass plays.
passing_tds	                                  	                The number of passing touchdowns.
passing_interceptions	                                  	    Number of passing interceptions
sacks_suffered	                                  	            Number of sacks taken as a QB
sack_yards_lost	                                  	            Yards lost from sacks suffered by this player
sack_fumbles	                                  	            The number of sacks suffered with a fumble.
        sack_fumbles_lost	                                    The number of sacks suffered with a lost fumble.
        passing_air_yards	                                  	Passing air yards (includes incomplete passes).
passing_yards_after_catch	                                  	Yards after the catch gained on plays in which player was the passer
passing_first_downs	                                  	First downs on pass attempts.
passing_epa	                                  	Total expected points added on pass attempts and sacks.
passing_cpoe	                                  	Completion percentage over expected for this player.
passing_2pt_conversions	                                  	Two-point conversion passes.
pacr	                                  	Passing (yards) Air (yards) Conversion Ratio - the number of passing yards per air yards thrown per game
carries	                                  	The number of official rush attempts (incl. scrambles and kneel downs). Rushes after a lateral reception don't count as carry.
rushing_yards	                                  	Yards gained when rushing with the ball (incl. scrambles and kneel downs). Also includes yards gained after obtaining a lateral on a play that started with a rushing attempt.
rushing_tds	                                  	The number of rushing touchdowns (incl. scrambles). Also includes touchdowns after obtaining a lateral on a play that started with a rushing attempt.
rushing_fumbles	                                  	The number of rushes with a fumble.
rushing_fumbles_lost	                                  	The number of rushes with a lost fumble.
rushing_first_downs	                                  	First downs on rush attempts (incl. scrambles).
rushing_epa	                                  	Expected points added on rush attempts (incl. scrambles and kneel downs).
rushing_2pt_conversions	                                  	Two-point conversion rushes
receptions	                                  	The number of pass receptions. Lateral receptions officially don't count as reception.
targets	                                  	The number of pass plays where the player was the targeted receiver.
receiving_yards	                                  	Yards gained after a pass reception. Includes yards gained after receiving a lateral on a play that started as a pass play.
receiving_tds	                                  	The number of touchdowns following a pass reception. Also includes touchdowns after receiving a lateral on a play that started as a pass play.
receiving_fumbles	                                  	The number of fumbles after a pass reception.
receiving_fumbles_lost	                                  	The number of fumbles lost after a pass reception.
receiving_air_yards	                                  	Receiving air yards (incl. incomplete passes).
receiving_yards_after_catch	                                  	Yards after the catch gained on plays in which player was receiver 
receiving_first_downs	                                  	Total number of first downs gained on receptions
receiving_epa	                                  	Total EPA on plays where this receiver was targeted
receiving_2pt_conversions	                                  	Two-point conversion receptions
racr	                                  	Receiving (yards) Air (yards) Conversion Ratio - the number of receiving yards per air yards targeted per game
target_share	                                  	Player's share of team receiving targets in this game
air_yards_share	                                  	Player's share of the team's air yards in this game
wopr	                                  	Weighted OPportunity Rating - 1.5 x target_share + 0.7 x air_yards_share - a weighted average that contextualizes total fantasy usage.
special_teams_tds	                                  	Total number of kick/punt return touchdowns
def_tackles_solo	                                  	Total number of solo tackles for this player
def_tackles_with_assist	                                  	Number of tackles this player had with an assisted tackle
def_tackle_assists	                                  	Number of assisted tackles for this player
def_tackles_for_loss	                                  	Number of tackles for loss (TFL) for this player
def_tackles_for_loss_yards	                                  	Yards lost from TFLs involving this player
def_fumbles_forced	                                  	Number of times a fumble was forced from this player
def_sacks	                                  	Number of sacks form this player
def_sack_yards	                                  	Yards lost from sacks forced by this player
def_qb_hits	                                  	Number of QB hits from this player (should not include plays where the QB was sacked)
def_interceptions	                                  	Number of interceptions forced by this player
def_interception_yards	                                  	yards gained/lost by interception returns from this player
def_pass_defended	                                  	Number of passes defended/broken up by this player
def_tds	                                  	Number of defensive touchdowns scored by this player
def_fumbles	                                  	Number of fumbles by this player
def_safeties	                                  	Number of safeties forced by this player
misc_yards	                                  	Miscellaneous yards attributed to this player
fumble_recovery_own	                                  	Number of the player's own team fumbles recovered
fumble_recovery_yards_own	                                  	Yards gained/lost on own fumble recoveries
fumble_recovery_opp	                                  	Number of the opponent's fumbles recovered
fumble_recovery_yards_opp	                                  	Yardage on opponent fumble recoveries
fumble_recovery_tds	                                  	Fumbles recovered and advanced for a touchdown
penalties	                                  	Number of penalties attributed to this player
penalty_yards	                                  	Penalty yardage on penalties attributed to this player
punt_returns	                                  	Count of punt returns by this player
punt_return_yards	                                  	Yards gained on punts returned by this player
kickoff_returns	                                  	Count of kick returns by this player
kickoff_return_yards	                                  	Yards gained on kick returns by this player
fg_made	                                  	Count of field goals made by this player
fg_att	                                  	Count of field goals attempted by this player
fg_missed	                                  	Count of field goals missed by this player
fg_blocked	                                  	Count of field goals attempted by this player that were blocked
fg_long	                                  	Longest successful field goal made by this player
fg_pct	                                  	Percentage of field goals successfully made
fg_made_0_19	                                  	Count of field goals within 0-19 yards made by this player
fg_made_20_29	                                  	Count of field goals within 20-29 yards made by this player
fg_made_30_39	                                  	Count of field goals within 30-39 yards made by this player
fg_made_40_49	                                  	Count of field goals within 40-49 yards made by this player
fg_made_50_59	                                  	Count of field goals within 50-59 yards made by this player
fg_made_60_	                                  	Count of field goals over 60 yards made by this player
fg_missed_0_19	                                  	Count of field goals missed between 0-19 yards by this player
fg_missed_20_29	                                  	Count of field goals missed between 20-29 yards by this player
fg_missed_30_39	                                  	Count of field goals missed between 30-39 yards by this player
fg_missed_40_49	                                  	Count of field goals missed between 40-49 yards by this player
fg_missed_50_59	                                  	Count of field goals missed between 50-59 yards by this player
fg_missed_60_	                                  	Count of field goals missed over 60 yards by this player
fg_made_list	                                              	Comma-separated string listing lengths of field goals made
fg_missed_list	                                              	Comma-separated string listing lengths of field goals missed
fg_blocked_list	                                              	Comma-separated string listing lengths of field goals blocked
fg_made_distance	                                  	Total distance on field goals made
fg_missed_distance	                                	Total distance on field goals missed
fg_blocked_distance	                                	Total distance on field goals blocked
pat_made	                                	Count of extra point kicks made
pat_att	                                	Count of extra point kicks attempted
pat_missed	                                	Count of extra point kicks missed
pat_blocked	                                	Count of extra point kicks blocked
pat_pct	                                	Percentage of extra point kicks successfully completed
gwfg_made	                                	Count of game winning field goals made
gwfg_att	                                	Count of game winning field goals attempted
gwfg_missed	                                	Count of game winning field goals missed
gwfg_blocked	                                	Count of game winning field goals blocked
gwfg_distance	                                	Total distance on game winning field goals completed
fantasy_points	                                	Standard fantasy points.
fantasy_points_ppr	                                	PPR fantasy points.