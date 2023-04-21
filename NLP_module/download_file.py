import pandas as pd
import os
import git

proj = pd.read_excel("dataset/projects.xlsx")
proj_df = proj.drop(['Stars', 'Contributors','Commits','Validated'], axis=1)
path_proj = os.path.join("..", "projects")
print(path_proj)
for i in range(15):
    # Clona la repository nella directory specificata
    print(proj_df.loc[i, 'Progetto'].strip())
    repo_dir = os.path.join(path_proj, proj_df.loc[i, 'Progetto'].strip())
    repo_url = proj_df.loc[i,'URL'].strip()
    repo = git.Repo.clone_from(repo_url, repo_dir)

    # Accede al tag specificato
    tag_name = proj_df.loc[i,'TAG'].strip()
    try:
        repo.git.checkout(tag_name)
    except git.exc.GitCommandError:
        # Identifica il conflitto nei file
        repo.git.status()
        
        # Annulla le modifiche apportate ai file conflittuali
        repo.git.restore("--source=HEAD", ":/")
        
        # Riprova il checkout del tag
        repo.git.checkout(tag_name)
    
    print("fine")

path = "../projects"
projects = []

# Itera su tutti i file e le cartelle nella cartella specificata
for item in os.listdir(path):
    # Se l'elemento nella cartella è una cartella, aggiungi il nome alla lista
    if os.path.isdir(os.path.join(path, item)):
        projects.append(item)

# Stampa la lista di nomi dei progetti
print(projects)