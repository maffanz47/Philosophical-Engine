# Philosophical Engine — Project Context Summary

## Project Objective

Transform the Philosophical Engine into a high-end, professional dashboard with a "Amber.io" inspired aesthetic, moving away from basic UI to a technical, data-dense interface.

## 1. Frontend Architecture

- **Framework:** React (Vite) located in `docs/frontend`.
- **Styling:** Custom CSS implementing a "Deep Obsidian" theme.
- **Design Tokens:**
  - Background: `#080808` with a 40px grid pattern.
  - Accent: `#f5b301` (Amber Gold).
  - Typography: `JetBrains Mono` for technical labels and `Inter` for primary data.
- **Layout:** A centralized "Terminal" style layout featuring a glowing title (`_PHILOSOPHICAL_TEXT_ENGINE`), model selection toggles, and dual-column inputs for text and file uploads.

## 2. Implemented Features

- **Inference Integration:** Connects to a Python FastAPI backend at `localhost:8000/predict`.
- **Tailored Metrics:**
  - **SVM (Deterministic):** Raw classification result.
  - **NN (Branch Tier 1):** Broad philosophical category with confidence percentage.
  - **NN (School Tier 2):** Specific school of thought with probability tracking.
  - **Complexity Score:** 0-10 index of text abstractness.
  - **The Librarian:** Horizontal scrolling book recommendations.
- **Visualization:** HTML5 Canvas-based scatter plot mapping text to unsupervised clusters using PCA coordinates.
- **File Handling:** Functional `.txt` file reader that populates the input area.

## 3. Current Pipeline Modifications (Phase 2)

We are currently splitting the pipeline into "Slow" and "Fast" tracks:

- **Thinking (Slow):** The current standard models trained on the full dataset.
- **Fast:** New SVM and K-Means models trained on a random 25% subset of the training data for rapid inference.
- **UI Interaction:** Added a Radio Button toggle (Fast vs. Slow) to the input dashboard to allow users to choose the model speed.

## 4. Pending Tasks

- **Discord Integration:** add a Discord Webhook notificationalong with the existing legacy email process.
- **Automated Reporting:** The notification will include evaluation metrics for all 4 models (SVM Slow/Fast, K-Means Slow/Fast) upon training completion.
- **Backend Routing:** Ensuring the `/predict` endpoint correctly handles the `speed` parameter to engage the appropriate model artifacts.
