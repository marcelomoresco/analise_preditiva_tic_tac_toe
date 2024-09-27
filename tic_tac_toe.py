import pickle
import tkinter as tk
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

class TicTacToeModel:
    def __init__(self, max_games=500, model_type="random_forest"):
        self.current_player = "X"
        self.board = [" " for _ in range(9)]
        self.buttons = []
        self.game_data = []
        self.model = None
        self.games_played = 0
        self.max_games = max_games
        self.min_training_games = 100
        self.model_type = model_type
        self.window = None
        self.info_label = None
        self.accuracy = None

    def check_winner(self):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        for condition in win_conditions:
            if self.board[condition[0]] == self.board[condition[1]] == self.board[condition[2]] != " ":
                return self.current_player
        if " " not in self.board:
            return "EMPATE"
        return None

    def reset_game(self):
        self.current_player = "X"
        self.board = [" " for _ in range(9)]
        for button in self.buttons:
            button.config(text=" ", state=tk.NORMAL, bg='#fafafa')

    def record_data(self, move):
        self.game_data.append({
            "board": self.board[:],
            "move": move,
            "player": self.current_player
        })

    def auto_move(self):
        if self.model and len(self.game_data) > 10:
            board_state = [1 if v == "X" else -1 if v == "O" else 0 for v in self.board]
            predicted_move = self.model.predict([board_state])[0]
            if self.board[predicted_move] == " ":
                return predicted_move
        
        available_moves = [i for i in range(9) if self.board[i] == " "]
        # if self.board[4] == " ":
        #     return 4
        
        for i in available_moves:
            self.board[i] = self.current_player
            if self.check_winner() == self.current_player:
                self.board[i] = " "
                return i
            self.board[i] = " "
        
        for i in available_moves:
            opponent = "O" if self.current_player == "X" else "X"
            self.board[i] = opponent
            if self.check_winner() == opponent:
                self.board[i] = " "
                return i
            self.board[i] = " "
        
        return random.choice(available_moves)

    def play_turn(self):
        if self.check_winner():
            return
        
        move = self.auto_move()
        self.board[move] = self.current_player
        self.buttons[move].config(text=self.current_player, state=tk.DISABLED)

        self.record_data(move)

        winner = self.check_winner()
        if winner:
            self.games_played += 1
            self.update_info_label()
            self.reset_game()

            if self.games_played >= self.max_games:
                self.train_model()
                print(f"finalizou treinamento {self.max_games} jogos.")
            else:
                self.window.after(10, self.play_turn)
            return

        self.current_player = "O" if self.current_player == "X" else "X"
        self.window.after(10, self.play_turn)

    def train_model(self):
        df = pd.DataFrame(self.game_data)
        X = df['board'].apply(lambda x: [1 if v == "X" else -1 if v == "O" else 0 for v in x]).tolist()
        y = df['move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf, params, model_filename = self.get_model_and_params()

        print("\nAntes do ajuste de hiperparâmetros:")
        self.evaluate_model_classifier(clf, X_train, y_train, X_test, y_test, "antes do ajuste")

        # Ajuste de hiperparâmetros com GridSearchCV
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Hiperparâmetros: {grid_search.best_params_}")

        # Modelo otimizado
        self.model = grid_search.best_estimator_
        self.accuracy = self.evaluate_model_classifier(self.model, X_train, y_train, X_test, y_test, "após o ajuste")

        with open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Modelo salvo como '{model_filename}'")

    def get_model_and_params(self):
        if self.model_type == "random_forest":
            clf = RandomForestClassifier()
            params = {
               'n_estimators': [50, 100],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5]
            }
            model_filename = 'random_forest_model.pkl'
        elif self.model_type == "decision_tree":
            clf = DecisionTreeClassifier()
            params = {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            }
            model_filename = 'decision_tree_model.pkl'
        return clf, params, model_filename

    def evaluate_model_classifier(self, model, X_train, y_train, X_test, y_test, stage):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\nDesempenho {stage} (Classificação):")
        print(f"Acurácia: {accuracy * 100:.2f}%")
        print(f"Precisão: {precision * 100:.2f}%")
        print(f"Revocação: {recall * 100:.2f}%")
        print(f"F1-Score: {f1 * 100:.2f}%")
        
        return accuracy

    def create_window(self):
        self.window = tk.Tk()
        self.window.title("Jogo da Velha - Machine Learning")
        self.window.geometry("600x700")
        self.window.configure(bg='#FFD700')

        button_font = ("Helvetica", 60, "bold")
        button_style = {
            'font': button_font, 'width': 4, 'height': 2,
            'state': tk.NORMAL, 'bg': '#fafafa', 'fg': '#000000', 'bd': 5, 'relief': 'raised'
        }

        for i in range(9):
            button = tk.Button(self.window, text=" ", **button_style)
            button.grid(row=i // 3, column=i % 3, sticky="nsew")
            self.buttons.append(button)

        for i in range(3):
            self.window.grid_columnconfigure(i, weight=1)
            self.window.grid_rowconfigure(i, weight=1)

        self.info_label = tk.Label(self.window, text=f"Treinando... Jogos: {self.games_played}/{self.max_games}", font=("Helvetica", 16))
        self.info_label.grid(row=3, column=0, columnspan=3, pady=20)

        self.window.after(10, self.play_turn)
        self.window.mainloop()

    def update_info_label(self):
        accuracy_text = f" Acurácia: {self.accuracy * 100:.2f}%" if self.accuracy else ""
        self.info_label.config(text=f"Treinando... Jogos: {self.games_played}/{self.max_games}{accuracy_text}")

if __name__ == "__main__":
    model_choice = "decision_tree"  #'decision_tree' "random_forest"
    game = TicTacToeModel(max_games=700, model_type=model_choice)
    game.create_window()
