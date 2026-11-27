# AI Disclosure Statement — EgoSight XR

**Project:** EgoSight XR
**Team:** Rajasthani  
**Institution:** Vellore Institute of Technology  
**Competition:** Samsung R&D Institute India XR Hackathon 2026  
**Track:** Track 4 — Action-Aware Egocentric Video Understanding and Intelligent Summarization

## 1. Use of Artificial Intelligence

EgoSight XR is an AI-powered video understanding system designed to analyze first-person and egocentric video footage. The system uses machine learning and computer vision models to identify broad activities, recognize supported temporal actions, and identify important video segments.

The system incorporates the following AI components:

- **Activity Classification:** A pretrained R3D-18 video recognition model, using Kinetics-400 knowledge, to classify broad activities.
- **Temporal Action Recognition:** A pretrained Temporal Segment Network (TSN) with ResNet-50, leveraging EPIC-KITCHENS-trained verb and noun recognition for supported actions.
- **Highlight Extraction:** A model-driven temporal selection process that identifies potentially important segments based on activity-related model confidence.
- **Structured Output:** Predictions are organized into machine-readable JSON containing activity labels, confidence scores, temporal action segments, and highlights.

## 2. Pretrained Models and Data

EgoSight XR uses pretrained models rather than claiming to train a foundation model from scratch.

The underlying models leverage knowledge learned from their respective training datasets, including Kinetics-400 and EPIC-KITCHENS.

The system applies these models to uploaded videos and generates predictions based on their learned representations. It does not claim that every activity or action can be recognized with equal accuracy.

## 3. AI-Assisted Development

AI-assisted development tools were used during the engineering process to support coding, debugging, documentation, testing.

These tools assisted the development workflow; they are not represented as the core video-understanding engine of EgoSight XR.

The implementation, integration, testing, and evaluation of the project were carried out as part of the team's development process.

## 4. Limitations and Transparency

EgoSight XR is a developing video-understanding system. Its predictions are model-generated estimates and may contain classification errors or miss certain actions.

Detailed verb/noun recognition is currently limited to the action vocabulary supported by the underlying trained model. Unsupported actions should not be presented as confidently recognized actions.

Highlights represent model-selected potentially important moments and should not be interpreted as guaranteed ground-truth event detection.

The system does not claim perfect accuracy, universal action recognition, or human-level video understanding.

## 5. Data Integrity and Responsible AI

The system is designed to generate predictions from video inputs rather than reproduce ground-truth annotations.

Evaluation annotations are not intended to be used as hidden-test prediction inputs or to tune the system against undisclosed evaluation labels.

We aim to maintain transparency regarding model capabilities, limitations, prediction confidence, and the distinction between recognized actions and unsupported activities.

**Declaration:** EgoSight XR uses pretrained artificial intelligence and computer vision models as core components of its video-understanding pipeline. AI-assisted tools were also used during development. The team remains responsible for the system's implementation, integration, testing, and submitted work.
