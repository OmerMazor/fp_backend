from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from datetime import datetime
import undetected_chromedriver as uc


def teams_data(home_team, away_team, home_market_value, away_market_value):    
    firefox_options = FirefoxOptions()
    # firefox_options.add_argument("--headless")

    # driver = webdriver.Firefox(options=firefox_options)
    # webgl_status = driver.execute_script("return !!window.WebGLRenderingContext;")
    # print(f"WebGL enabled: {webgl_status}")
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")
    driver = uc.Chrome(options=options) 
    url = f"https://fbref.com/en/search/search.fcgi?hint={home_team.replace(' ', '+')}&search={home_team.replace(' ', '+')}&pid=&idx="
    driver.get(url)
    time.sleep(5)
    home_id = ""
    away_id = ""
    try:
        time.sleep(3)  # חכה 3 שניות לפני החיפוש* 
        driver.find_element(By.ID, "clubs-tab").click() 
        link = driver.find_element(By.XPATH, "//*[@id='clubs']/div[1]/div[2]/a") 
        home_id = link.text.split('/')[3]
    except:
        pass
    url = f"https://fbref.com/en/search/search.fcgi?hint={away_team.replace(' ', '+')}&search={away_team.replace(' ', '+')}&pid=&idx="
    driver.get(url)
    try:
        time.sleep(3)  # חכה 3 שניות לפני החיפוש* 
        driver.find_element(By.ID, "clubs-tab").click() 
        link = driver.find_element(By.XPATH, "//*[@id='clubs']/div[1]/div[2]/a") 
        away_id = link.text.split('/')[3]
    except:
        pass

    url = f"https://fbref.com/en/stathead/matchup/teams/{home_id}/{away_id}/"
    driver.get(url)
    # המתנה לטעינת אלמנט מסוים
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "content")))
    except Exception as e:
        pass

    home_wins = ""
    draws = ""
    away_wins = ""
    try:
        home_wins = driver.find_element("xpath", "//div[@class='scorebox']/div[1]/div[2]").text.split()[0]
        draws = driver.find_element("xpath", "//div[@class='scorebox']/div[1]/div[3]").text.split()[0]
        away_wins = driver.find_element("xpath", "//div[@class='scorebox']/div[2]/div[2]").text.split()[0]
    except Exception as e:
        pass




    home_hosting = "h"
    second_url = f"https://fbref.com/en/squads/{home_id}"
    driver.get(second_url)
    home_position = "NA"
    home_games = 0
    try:
        # מציאת האלמנט שמכיל את המידע
        record_element = driver.find_element("xpath", "//p[strong[text()='Record:']]")
        home_position = record_element.text.split(" ")[4].replace("(", "")
        home_games = eval(record_element.text.split(" ")[1].replace(",","").replace("-","+"))
    except:
        pass

    # המתן לטעינת הדף
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "p")))
    home_goals = "NA"
    home_goals_against = "NA"
    try:
        home_goals_text = driver.find_element("xpath", "//p[contains(., 'Goals:')]").text
        home_goals = home_goals_text.split("Goals:")[1].split(",")[0].split(" ")[2].replace("(", "")
        home_goals_against_text = driver.find_element("xpath", "//p[contains(., 'Goals Against:')]").text
        home_goals_against = home_goals_against_text.split("Goals Against:")[1].split(",")[0].split(" ")[2].replace("(","")
    except:
        pass
    home_goalkeeping = "NA"
    try:
        home_goalkeeping = driver.find_element("xpath", "//tfoot/tr/td[@data-stat='gk_save_pct']").text
    except:
        pass
    home_yellow_cards = "NA"
    try:
        home_yellow_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_yellow']").text + "/" + str(home_games)), 3)
    except:
        pass
    home_red_cards = "NA"
    try:
        home_red_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_red']").text + "/" + str(home_games)), 3)
    except:
        pass

    home_defense = 'NA'
    try:
        home_defense = driver.find_element("xpath",
                                            "//table/ tfoot / tr[1] / td[@data-stat='challenge_tackles_pct']").text
    except:
        pass

    home_shot_on_target = "NA"
    try:
        home_shot_on_target = driver.find_element("xpath",
                                                    "// table/ tfoot / tr[1] / td[@data-stat='shots_on_target_per90']").text
    except:
        pass

    home_accuracy = "NA"
    try:
        home_accuracy = driver.find_element("xpath",
                                                    "// table/ tfoot / tr[1] / td[@data-stat='goals_per_shot_on_target']").text
    except:
        pass

    home_corners = "NA"
    try:
        home_corners = round(eval(driver.find_element("xpath",
                                            "// table / tfoot / tr[1] / td[@data-stat='corner_kicks']").text + "/" + str(home_games)),3)

    except:
        pass



    away_hosting = "a"
    third_url = f"https://fbref.com/en/squads/{away_id}"
    driver.get(third_url)
    away_position = "NA"
    away_games = 0
    try:
        # מציאת האלמנט שמכיל את המידע
        record_element = driver.find_element("xpath", "//p[strong[text()='Record:']]")
        away_position = record_element.text.split(" ")[4].replace("(", "")
        away_games = eval(record_element.text.split(" ")[1].replace(",", "").replace("-", "+"))
    except:
        pass

    # המתן לטעינת הדף
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "p")))

    away_goals = "NA"
    away_goals_against = "NA"
    try:
        away_goals_text = driver.find_element("xpath", "//p[contains(., 'Goals:')]").text
        away_goals = away_goals_text.split("Goals:")[1].split(",")[0].split(" ")[2].replace("(","")
        away_goals_against_text = driver.find_element("xpath", "//p[contains(., 'Goals Against:')]").text
        away_goals_against = away_goals_against_text.split("Goals Against:")[1].split(",")[0].split(" ")[2].replace("(","")
    except:
        pass
    away_goalkeeping = "NA"
    try:
        away_goalkeeping = driver.find_element("xpath", "//tfoot/tr/td[@data-stat='gk_save_pct']").text
    except:
        pass
    away_yellow_cards = "NA"
    try:
    # שליפת כרטיסים צהובים מהשורה התחתונה
        away_yellow_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_yellow']").text + "/" + str(away_games)), 3)
    except:
        pass

    away_red_cards = "NA"
    try:
    # שליפת כרטיסים אדומים מהשורה התחתונה
        away_red_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_red']").text + "/" + str(away_games)), 3)
    except:
        pass

    away_defense = 'NA'
    try:
        away_defense = driver.find_element("xpath",
                                            "//table/ tfoot / tr[1] / td[@data-stat='challenge_tackles_pct']").text
        pass
    except:
        pass

    away_shot_on_target = "NA"
    try:
        away_shot_on_target = driver.find_element("xpath",
                                                    "// table / tfoot / tr[1] / td[@data-stat='shots_on_target_per90']").text

    except:
        pass
    away_accuracy = "NA"
    try:
        away_accuracy = driver.find_element("xpath",
                                                    "// table/ tfoot / tr[1] / td[@data-stat='goals_per_shot_on_target']").text
    except:
        pass
    away_corners = "NA"
    try:

        away_corners = round(eval(driver.find_element("xpath",
                                            "// table / tfoot / tr[1] / td[@data-stat='corner_kicks']").text + "/" + str(away_games)),3)

    except:
        pass

    driver.quit()
    return home_wins, draws, away_wins, home_position, home_goals, home_goals_against, \
    home_accuracy, home_goalkeeping, home_red_cards, home_shot_on_target, away_position, \
    away_goals, away_goals_against, away_shot_on_target, away_accuracy, away_goalkeeping, home_games, away_games

def matches_day_data():

    firefox_options = FirefoxOptions()
    # firefox_options.add_argument("--headless")

    driver = webdriver.Firefox(options=firefox_options)
    webgl_status = driver.execute_script("return !!window.WebGLRenderingContext;")
    print(f"WebGL enabled: {webgl_status}")
    today = datetime.today().strftime("%d-%m-%Y")
    url = f"https://fbref.com/en/matches/{today}"
    driver.get(url)
    driver.get(url)
    try:
        home_team_cells = driver.find_elements("xpath", "//td[@data-stat='home_team']/a")
        away_team_cells = driver.find_elements("xpath", "//td[@data-stat='away_team']/a")
        clubs = []

        for home_team_cell, away_team_cell in zip(home_team_cells, away_team_cells):
            home_full_href = home_team_cell.get_attribute("href")  
            home_relevant_part = home_full_href.split("/en/squads/")[1]  
            away_full_href = away_team_cell.get_attribute("href")
            away_relevant_part = away_full_href.split("/en/squads/")[1]
            clubs.append([home_relevant_part, away_relevant_part])
        for m in clubs:
            url = f"https://fbref.com/en/stathead/matchup/teams/{m[0].split('/')[0]}/{m[1].split('/')[0]}/{m[0].split('/')[1]}-vs-{m[1].split('/')[1]}-History"
            driver.get(url)
            teams = f"{m[0].split('/')[1].replace('-Stats', '')}-vs-{m[1].split('/')[1].replace('-Stats', '')}"
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "content")))
            except Exception as e:
                pass

            home_wins = ""
            draws = ""
            away_wins = ""
            try:
                home_wins = driver.find_element("xpath", "//div[@class='scorebox']/div[1]/div[2]").text.split()[0]
                draws = driver.find_element("xpath", "//div[@class='scorebox']/div[1]/div[3]").text.split()[0]
                away_wins = driver.find_element("xpath", "//div[@class='scorebox']/div[2]/div[2]").text.split()[0]
            except Exception as e:
                pass

            home_hosting = "h"
            second_url = f"https://fbref.com/en/squads/{m[0]}"
            driver.get(second_url)
            home_position = "NA"
            home_games = 0
            try:
                # מציאת האלמנט שמכיל את המידע
                record_element = driver.find_element("xpath", "//p[strong[text()='Record:']]")
                home_position = record_element.text.split(" ")[4].replace("(", "")
                home_games = eval(record_element.text.split(" ")[1].replace(",","").replace("-","+"))
            except:
                pass

            # המתן לטעינת הדף
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "p")))
            home_goals = "NA"
            home_goals_against = "NA"
            try:
                home_goals_text = driver.find_element("xpath", "//p[contains(., 'Goals:')]").text
                home_goals = home_goals_text.split("Goals:")[1].split(",")[0].split(" ")[2].replace("(", "")
                home_goals_against_text = driver.find_element("xpath", "//p[contains(., 'Goals Against:')]").text
                home_goals_against = home_goals_against_text.split("Goals Against:")[1].split(",")[0].split(" ")[2].replace("(","")
            except:
                pass
            home_goalkeeping = "NA"
            try:
                home_goalkeeping = driver.find_element("xpath", "//tfoot/tr/td[@data-stat='gk_save_pct']").text
            except:
                pass
            home_yellow_cards = "NA"
            try:
                home_yellow_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_yellow']").text + "/" + str(home_games)), 3)
            except:
                pass
            home_red_cards = "NA"
            try:
                home_red_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_red']").text + "/" + str(home_games)), 3)
            except:
                pass

            home_defense = 'NA'
            try:
                home_defense = driver.find_element("xpath",
                                                   "//table/ tfoot / tr[1] / td[@data-stat='challenge_tackles_pct']").text
            except:
                pass

            home_shot_on_target = "NA"
            try:
                home_shot_on_target = driver.find_element("xpath",
                                                          "// table/ tfoot / tr[1] / td[@data-stat='shots_on_target_per90']").text
            except:
                pass

            home_accuracy = "NA"
            try:
                home_accuracy = driver.find_element("xpath",
                                                          "// table/ tfoot / tr[1] / td[@data-stat='goals_per_shot_on_target']").text
            except:
                pass

            home_corners = "NA"
            try:
                home_corners = round(eval(driver.find_element("xpath",
                                                   "// table / tfoot / tr[1] / td[@data-stat='corner_kicks']").text + "/" + str(home_games)),3)

            except:
                pass

            home_market_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={m[0].split('/')[1].replace('-Stats', '').replace('-', '+')}"
            driver.get(home_market_url)
            home_market_value = "NA"
            try:
                WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "yw0")))
            except Exception as e:
                pass
            try:

                market_value_element = driver.find_element(
                    "xpath",
                    f"//tr[td[@class='zentriert suche-vereinswappen']/img[@title='{m[0].split('/')[1].replace('-Stats', '').replace('-', ' ')}']]/td[@class='rechts']"
                )
                home_market_value = market_value_element.text
            except:
                pass

            away_hosting = "a"
            third_url = f"https://fbref.com/en/squads/{m[1]}"
            driver.get(third_url)
            away_position = "NA"
            away_games = 0
            try:
                record_element = driver.find_element("xpath", "//p[strong[text()='Record:']]")
                away_position = record_element.text.split(" ")[4].replace("(", "")
                away_games = eval(record_element.text.split(" ")[1].replace(",", "").replace("-", "+"))
            except:
                pass
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "p")))

            away_goals = "NA"
            away_goals_against = "NA"
            try:
                away_goals_text = driver.find_element("xpath", "//p[contains(., 'Goals:')]").text
                away_goals = away_goals_text.split("Goals:")[1].split(",")[0].split(" ")[2].replace("(","")
                away_goals_against_text = driver.find_element("xpath", "//p[contains(., 'Goals Against:')]").text
                away_goals_against = away_goals_against_text.split("Goals Against:")[1].split(",")[0].split(" ")[2].replace("(","")
            except:
                pass
            away_goalkeeping = "NA"
            try:
                away_goalkeeping = driver.find_element("xpath", "//tfoot/tr/td[@data-stat='gk_save_pct']").text
            except:
                pass
            away_yellow_cards = "NA"
            try:
                away_yellow_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_yellow']").text + "/" + str(away_games)), 3)
            except:
                pass

            away_red_cards = "NA"
            try:
                away_red_cards = round(eval(driver.find_element("xpath", "//tfoot/tr/td[@data-stat='cards_red']").text + "/" + str(away_games)), 3)
            except:
                pass

            away_defense = 'NA'
            try:
                away_defense = driver.find_element("xpath",
                                                   "//table/ tfoot / tr[1] / td[@data-stat='challenge_tackles_pct']").text
                pass
            except:
                pass

            away_shot_on_target = "NA"
            try:
                away_shot_on_target = driver.find_element("xpath",
                                                          "// table / tfoot / tr[1] / td[@data-stat='shots_on_target_per90']").text

            except:
                pass
            away_accuracy = "NA"
            try:
                away_accuracy = driver.find_element("xpath",
                                                          "// table/ tfoot / tr[1] / td[@data-stat='goals_per_shot_on_target']").text
            except:
                pass
            away_corners = "NA"
            try:

                away_corners = round(eval(driver.find_element("xpath",
                                                   "// table / tfoot / tr[1] / td[@data-stat='corner_kicks']").text + "/" + str(away_games)),3)

            except:
                pass

            away_market_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={m[1].split('/')[1].replace('-Stats', '').replace('-', '+')}"
            driver.get(away_market_url)
            try:
                WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "yw0")))
            except Exception as e:
                pass
            away_market_value = "NA"
            try:
                market_value_element = driver.find_element(
                    "xpath",
                    f"//tr[td[@class='zentriert suche-vereinswappen']/img[@title='{m[1].split('/')[1].replace('-Stats', '').replace('-', ' ')}']]/td[@class='rechts']"
                )
                away_market_value = market_value_element.text
            except:
                pass





            print(home_wins, draws, away_wins, home_hosting, home_games, home_position, home_goals, home_goals_against, home_goalkeeping, home_yellow_cards,
                  home_red_cards, home_defense, home_shot_on_target, home_accuracy, home_corners, away_hosting, away_games,
                  away_position, away_goals, away_goals_against, away_goalkeeping, away_yellow_cards, away_red_cards, away_defense,
                  away_shot_on_target, away_accuracy, away_corners,home_market_value, away_market_value, teams)




    except Exception as e:
        print(f"An error occurred: {e}")
    driver.quit()