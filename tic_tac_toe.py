import pickle
import tkinter as tk
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
import pandas as pd

class TicTacToeModel:
    def __init__(self, max_games=10, model_type="random_forest"):
        self.current_player = "X"
        self.board = [" " for _ in range(9)]
        self.buttons = []
        self.game_data = []
        self.model = None  # Modelo para prever jogadas
        self.games_played = 0
        self.max_games = max_games
        self.min_training_games = 50
        self.model_type = model_type  # "random_forest", "decision_tree", "decision_tree_regressor"
        self.window = None

    def check_winner(self):
        """Verifica as condições de vitória."""
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
        """Reseta o tabuleiro para iniciar um novo jogo."""
        self.current_player = "X"
        self.board = [" " for _ in range(9)]
        for button in self.buttons:
            button.config(text=" ", state=tk.NORMAL, bg='#fafafa')

    def record_data(self, move):
        """Grava o estado do tabuleiro e a jogada atual."""
        self.game_data.append({
            "board": self.board[:],
            "move": move,
            "player": self.current_player
        })

    def auto_move(self):
        """Escolhe o movimento automaticamente, usando o modelo treinado ou regras básicas."""
        if self.model and len(self.game_data) > 10:
            board_state = [1 if v == "X" else -1 if v == "O" else 0 for v in self.board]
            predicted_move = self.model.predict([board_state])[0]
            if self.board[predicted_move] == " ":
                return predicted_move
            else:
                available_moves = [i for i in range(9) if self.board[i] == " "]
                return random.choice(available_moves)
        else:
            available_moves = [i for i in range(9) if self.board[i] == " "]
            
            # pega o centro
            if self.board[4] == " ":
                return 4
            
            # bloquear o oponente ou vencer
            for i in available_moves:
                self.board[i] = self.current_player
                if self.check_winner() == self.current_player:
                    self.board[i] = " "
                    return i
                self.board[i] = " "
            
            # aleatória se não houver estratégia óbvia
            return random.choice(available_moves)

    def play_turn(self):
        """Realiza o turno do jogador atual."""
        if self.check_winner():
            return
        
        move = self.auto_move()
        self.board[move] = self.current_player
        self.buttons[move].config(text=self.current_player, state=tk.DISABLED)

        self.record_data(move)

        winner = self.check_winner()
        if winner:
            self.games_played += 1
            print(f"Jogo {self.games_played} concluído.")
            self.reset_game()

            if self.games_played >= self.max_games:
                self.train_model()
                print(f"Treinamento concluído após {self.max_games} jogos.")
            else:
                self.window.after(1000, self.play_turn)
            return

        # Troca o jogador
        self.current_player = "O" if self.current_player == "X" else "X"
        self.window.after(1000, self.play_turn)

    def train_model(self):
        """Treina o modelo após atingir o número de jogos necessários."""
        if len(self.game_data) < self.min_training_games:
            print("Jogos insuficientes.")
            return

        df = pd.DataFrame(self.game_data)
        X = df['board'].apply(lambda x: [1 if v == "X" else -1 if v == "O" else 0 for v in x]).tolist()
        y = df['move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Escolha do modelo
        clf, params, model_filename = self.get_model_and_params()

        # Avaliação inicial
        if self.model_type == "decision_tree_regressor":
            self.evaluate_model_regressor(clf, X_train, y_train, X_test, y_test, "antes do ajuste")
        else:
            self.evaluate_model_classifier(clf, X_train, y_train, X_test, y_test, "antes do ajuste")

        # Ajuste dos parâmetros com RandomizedSearchCV
        randomized_search = RandomizedSearchCV(clf, param_distributions=params, cv=2, n_iter=5, n_jobs=-1)
        randomized_search.fit(X_train, y_train)

        print(f"Hiperparâmetros: {randomized_search.best_params_}")

        # salva modelo otimizado
        self.model = randomized_search.best_estimator_

        if self.model_type == "decision_tree_regressor":
            self.evaluate_model_regressor(self.model, X_train, y_train, X_test, y_test, "após o ajuste")
        else:
            self.evaluate_model_classifier(self.model, X_train, y_train, X_test, y_test, "após o ajuste")

        with open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Modelo salvo como '{model_filename}'")

    def get_model_and_params(self):
        """Define o modelo e os parâmetros de hiperparâmetros baseados no tipo de modelo."""
        if self.model_type == "random_forest":
            clf = RandomForestClassifier()
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
            model_filename = 'random_forest_model.pkl'
        elif self.model_type == "decision_tree":
            clf = DecisionTreeClassifier()
            params = {
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
            model_filename = 'decision_tree_model.pkl'
        elif self.model_type == "decision_tree_regressor":
            clf = DecisionTreeRegressor()
            params = {
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
            model_filename = 'decision_tree_regressor.pkl'
        return clf, params, model_filename

    def evaluate_model_classifier(self, model, X_train, y_train, X_test, y_test, stage):
        """Avalia o modelo de classificação usando várias métricas."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="micro")
        recall = recall_score(y_test, y_pred, average="micro")
        f1 = f1_score(y_test, y_pred, average="micro")

        print(f"\nDesempenho {stage} (Classificação):")
        print(f"Acurácia: {accuracy * 100:.2f}%")
        print(f"Precisão: {precision * 100:.2f}%")
        print(f"Revocação: {recall * 100:.2f}%")
        print(f"F1-Score: {f1 * 100:.2f}%")

    def evaluate_model_regressor(self, model, X_train, y_train, X_test, y_test, stage):
        """Avalia o modelo de regressão usando MSE e R2 Score."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nDesempenho {stage} (Regressão):")
        print(f"Erro quadrático médio (MSE): {mse:.4f}")
        print(f"Coeficiente de determinação (R2 Score): {r2:.4f}")

    def create_window(self):
        """Cria a janela gráfica do jogo da velha."""
        self.window = tk.Tk()
        self.window.title("Jogo da Velha - Machine Learning")
        self.window.geometry("600x600")
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

        # redimensiona colun
        for i in range(3):
            self.window.grid_columnconfigure(i, weight=1)
            self.window.grid_rowconfigure(i, weight=1)

        self.window.after(1000, self.play_turn)
        self.window.mainloop()


if __name__ == "__main__":
    model_choice = "random_forest"  # ou "decision_tree" ou "decision_tree_regressor"
    game = TicTacToeModel(max_games=100, model_type=model_choice)
    game.create_window()
