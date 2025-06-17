# Guide de construction pour CNC dans WEKA

## Prérequis
- Apache Ant installé
- Version de Java compatible avec votre installation WEKA

## Étapes de construction

### 1. Configuration des composants
- Copiez tous les éléments nécessaires du dossier `FCA` et le classificateur (dans le dossier `rules`) depuis votre espace de travail vers cette architecture
- Assurez-vous que la structure du projet respecte l'organisation standard de WEKA

### 2. Compilation du package
- Exécutez la commande `ant clean package` dans le terminal à la racine du projet
- Cette commande compilera toutes vos dépendances et créera l'archive d'installation

### 3. Installation dans WEKA
- Vous trouverez le fichier `CNC.zip` dans le dossier `dist`
- Importez ce fichier dans WEKA via Tools → Package Manager → File/URL
- Après installation, redémarrez WEKA pour utiliser le classificateur

## Résolution des problèmes courants
- Si vous rencontrez des erreurs de version Java, vérifiez que vous compilez avec une version compatible avec votre installation WEKA
- Pour les erreurs de dépendances, assurez-vous que votre fichier Description.props est correctement formaté

### Note
Dépendamment de la machine, il faut spécifier l'emplacement de weka.jar dans build.xml, vous devrez potentiellement le changer en fonction de votre configuration
