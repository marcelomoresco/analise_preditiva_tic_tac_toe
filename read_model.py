import pickle

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Hiperparâmetros do modelo:")
print(model.get_params())

if hasattr(model, 'n_estimators'):
    print(f"Número de árvores no modelo: {model.n_estimators}")
    print(f"Quantidade de árvores na floresta: {len(model.estimators_)}")

elif hasattr(model, 'tree_'):
    print(f"Profundidade da árvore: {model.tree_.max_depth}")
    print(f"Número de nós terminais: {model.tree_.n_leaves}")
    print(f"Número total de nós: {model.tree_.node_count}")

if hasattr(model, 'feature_importances_'):
    print(f"Importância das features: {model.feature_importances_}")

if hasattr(model, 'estimators_'):
    first_tree = model.estimators_[0]
    print(f"Profundidade da primeira árvore: {first_tree.tree_.max_depth}")
