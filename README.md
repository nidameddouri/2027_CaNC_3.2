# CNC – Classifier based on Nominal Concepts (Weka Plugin)

## 📖 Overview
CNC (Classifier Nominal Concepts) is a Weka plugin based on **Formal Concept Analysis (FCA)** for rule-based classification.  
It extracts attribute closures using FCA and generates classification rules according to different strategies.  
This plugin is designed to extend Weka with an interpretable, FCA-driven rule classifier.

---

## 📂 Project Structure

```
2025_canc_3.0/
│── package_build/
│   └── src/
│       └── main/
│           └── java/
│               └── weka/
│                   └── classifiers/
│                       ├── fca/
│                       │   ├── _GaloisConnection.java
│                       │   ├── _Rules.java
│                       │   ├── Classification_Rule.java
│                       │   ├── Classify_Instance.java
│                       │   ├── JsonExporter.java
│                       │   ├── RuleExtractor.java
│                       │   └── TestClassificationRule.java
│                       └── rules/
│                           └── CNC.java
│
│── weka-src/
│   ├── src/main/java/     # Weka main source (optional, for dev)
│   └── src/test/java/     # Weka test source (optional, for dev)
│
│── build.xml              # Apache Ant build file (for compilation & packaging)
│── dist/                  # Generated jar and zip packages
│── README.md              # Main documentation (this file)
```

---

## 🛠️ Installation of Required Tools

### 🔹 Windows users

1. **Install [Weka](https://waikato.github.io/weka-wiki/downloading_weka/)**

2. **Install [JDK 17 for Windows](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html)**
   > CNC has been tested with JDK 17. Using earlier versions may cause compatibility issues.

3. **Install [Git for Windows](https://git-scm.com/download/win)** (for source code management)

4. **Install [Apache Ant](https://ant.apache.org/manual/install.html)** (for building and packaging the plugin)

5. Once tools are installed, proceed to [Installation](#-installation) to set up the CNC plugin.


### 🔹 Linux users

1. **Install [Weka](https://waikato.github.io/weka-wiki/downloading_weka/)**

2. **Install JDK 17:**
   ```bash
   sudo apt install openjdk-17-jdk
   ```

3. **Install Git:**
   ```bash
   sudo apt install git
   ```

4. **Install Ant:**
   ```bash
   sudo apt install ant
   ```

5. Once tools are installed, proceed to [Installation](#-installation) to set up the CNC plugin.

---

## ⚙️ Installation

### 🔹 Install for Weka Users

You have two options to install the CNC plugin in Weka:

#### Option 1: Install with Weka Package Manager (recommended)
1. Open Weka.  
2. Go to **Tools → Package Manager**.  
3. In the Package Manager window, click **File → Install New Package (from zip file)**.  
4. Select the CNC `.zip` package generated (e.g., `cnc-weka.zip`).  
5. Restart Weka.  
6. The CNC classifier will appear under:  
   ```
   Classifiers → rules → CNC
   ```

#### Option 2: Manual installation of the JAR file
1. Download the precompiled `cnc-weka.jar`.  
2. Copy it to your Weka packages directory:
   ```bash
   cp cnc-weka.jar $WEKA_HOME/packages/
   ```
3. Restart Weka.  
4. The CNC classifier will appear under:  
   ```
   Classifiers → rules → CNC
   ```

---

### 🔹 Install for Developers

If you want to **modify or extend CNC**, you need to compile it yourself.

#### Requirements
- **JDK 17** (tested with OpenJDK 17)
- **Apache Ant** (for build and packaging)
- Weka source code (optional, only if you want to debug/extend inside Weka core)

#### Steps
1. Clone or download the project:
   ```bash
   git clone https://gitlab.com/your-username/cnc-weka.git
   cd cnc-weka/package_build
   ```   
2. Modify the source code
3. Compile your modifications using **Ant**:
   ```bash
   ant command : ant -buildfile path_to_build.xml package -Dpackage=2025_canc_3.0
   ```

   This will:
   - Compile the Java sources.
   - Generate the `.jar`.
   - Create a `.zip` package compatible with Weka’s Package Manager.
   

4. Install the generated package:
   - Locate the `.zip` file in the `dist/` directory.
   - Open Weka’s **Package Manager**.
   - Install the `.zip` package.

   ## 🚀 Usage

### Command Line
```bash
java -cp weka.jar:cnc-weka.jar weka.classifiers.rules.CNC   -t dataset.arff   -THRESHOLD 0.25   -aggregationStrategy 1   -conceptStrategy 6   -measure gainratio
```

### Weka GUI
1. Open **Weka Explorer**.  
2. Go to the **Classify** tab.  
3. Select **rules → CNC** as the classifier.  
4. Configure the parameters:
   - Threshold  
   - Aggregation strategy  
   - Concept strategy  
   - Relevance measure  
   - Vote strategy   

---

## 🔧 Parameters

The CNC classifier offers several configurable parameters to control concept learning, attribute selection, and vote aggregation.

| Parameter                        | Description                                                                                                       | Example |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------|---------|
| `aboveUnderPertThreshold`        | Minimum relevance (in %) threshold for attribute selection (between 0 and 1). Used with Closure of attributes Above/Under threshold. | `0.25`  |
| `autoPertThresholdComb`          | If enabled, automatically determines the optimal threshold (used with Closure of attributes Above/Under threshold). | `true`  |
| `beforeAfterDisc`                | Defines when to apply the threshold: before or after discretization.                                               | `before` |
| `pertinenceMeasure`              | Measure used to evaluate attribute relevance (`gainratio`, `infoGain`, etc.).                                      | `gainratio` |
| `aggregationVoters`              | Strategy used to aggregate votes from selected concepts.                                                           | `weighted` |
| `closureStrategy`                | Closure strategy for concept learning.                                                                             | `6` |
| `policySelectionOccurence`       | Policy to select the final concept in case of conflicts.                                                           | `majority` |
| `topKVoters`                     | Number of top voters to keep in aggregation (used with Weighted Vote Aggregation With Top-K Best Voters).          | `5` |
| `pluralityVotersThreshold`       | Minimum percentage of agreement among voters to accept a decision (Plurality Vote Aggregation with Minimum Threshold). | `0.6` |
| `weightedVoteThreshold`          | Minimum weight required for a voter to be considered (Weighted Vote Aggregation with Minimum Weight Threshold).    | `0.15` |
| `fullAttribCombination`          | If enabled, tests all possible attribute combinations (all strategies).                                            | `false` |
| `attributesCombinSize`           | Size of attribute combinations (used with Closure of N attributes combination).                                    | `3` |
| `attributesCombinPercent`        | Percentage of attributes to select for each combination (Closure of Percentage attributes combination).             | `40` |
| `varianceAttriCombPercent`       | Variance applied around the selected attribute percentage (Closure of Percentage attributes combination + variance). | `10` |
| `vectorialConceptList`           | List of concepts in vector format to use directly.                                                                | `[...]` |
| `filter`                         | Filter to apply to the dataset before classification (e.g., discretization).                                       | `true` |
| `subContexts`                    | If enabled, splits dataset into sub-contexts by class value and extracts rules from each.                          | `true` |



## 📖 References

### Original Methodology
```latex
@book{ganter1999formal,
  title={Formal Concept Analysis: Mathematical Foundations},
  author={Ganter, Bernhard and Wille, Rudolf},
  year={1999},
  publisher={Springer}
}
```
---


## 👩‍💻 Developer & Supervisors

- **Developer:**  
  Developer of the CNC (Classifier Nominal Concepts) plugin for Weka  
  📧 Contact: [yasmine.souissi@epita.fr] , [souissiy35@gmail.com] 
  🔗 GitLab: [https://gitlab.com/YasmineSouissi](https://gitlab.com/YasmineSouissi)


- **Supervised by:**  
  **Nidà MEDDOURI** (nida.meddouri@epita.fr)  
  **Fabrice BOISSIER** (fabrice.boissier@epita.fr)