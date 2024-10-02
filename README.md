# Hibrid Sakk AI Tervezése és Implementációja

## Bevezetés

Ez a projekt egy hibrid sakk mesterséges intelligencia fejlesztésére irányul, amely konvolúciós neurális hálózatot (CNN), Monte Carlo Tree Search algoritmust (MCTS), nyitókönyvet és grafikus felhasználói felületet (GUI) integrál. Ez a mesterséges intelligencia képes a sakkjátszmák értékelésére és megoldására, felhasználva a mélytanulási technikák és keresési algoritmusok előnyeit.

### Fontos Megjegyzés

Ez a projekt egy diplomamunka része, ezért az itt található kód csak oktatási és kutatási célokra használható fel. A kereskedelmi felhasználás nem engedélyezett.

## Követelmények

A projekt futtatásához szükséges szoftverek és könyvtárak telepítése az alábbi lépésekben történik:

- Python 3.12
- Anaconda: Ajánlott a projekt futtatásához.

## Telepítés

### 1. Docker használatával:

Ha rendelkezésre áll Docker, a projekt futtatása nagyon egyszerű. A Docker használatához:

```bash
docker build -t sakk_ai .
docker run -p 5000:5000 sakk_ai
```

### 2. Kézi telepítés:

Először klónozza a projektet, majd hozza létre a környezetet a conda segítségével.

```bash
git clone <projekt-repo-url>
cd <project-helye>
conda env create -f environment.yml
conda activate hybrid_chess_ai
```

A környezet aktiválása után futtassa a fő szkriptet:

```bash
python main.py
```

## Használat

- Az AI mesterséges intelligencia ellen lehet játszani az interaktív sakkfelületen keresztül, amely drag-and-drop funkcióval rendelkezik.
- Az AI belső működésének megfigyelésére lehetőség van az MCTS keresés vizualizálásával és a lépések valószínűségeinek megjelenítésével.

## További Információk

Ez a projekt erősen épít a következő területekre:
- Konvolúciós Neurális Hálózatok
- Monte Carlo Tree Search
- Sakk nyitókönyvek

A részletes technikai dokumentáció megtalálható a docs/ mappában.