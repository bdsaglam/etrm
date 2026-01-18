## ARC-AGI-2 Leaderboard

ARC-AGI-1

ARC-AGI-2

Author:

All Authors

Model type:

All Types

Model:

All Models

### Understanding the Leaderboard

ARC-AGI has evolved from its first version (ARC-AGI-1) which measured basic fluid intelligence, to ARC-AGI-2 which challenges systems to demonstrate both high adaptability and high efficiency.

The scatter plot above visualizes the critical relationship between cost-per-task and performance - a key measure of intelligence efficiency. True intelligence isn't just about solving problems, but solving them efficiently with minimal resources.

### Interpreting the data

- **Reasoning Systems Trend Line** solutions display connected points representing the same model at different reasoning levels. These trend lines illustrate how increased reasoning time affects performance, typically showing asymptotic behavior as thinking time increases.
- **Base LLMs** solutions represent single-shot inference from standard language models like GPT-4.5 and Claude 3.7, without extended reasoning capabilities. These points demonstrate raw model performance without additional reasoning enhancements.
- **Kaggle Systems** solutions showcase competition-grade submissions from the Kaggle challenge, operating under strict computational constraints ($50 compute budget for 120 evaluation tasks). These represent purpose-built, efficient methods specifically designed for the ARC Prize.

For more information on our reporting process, see our [testing policy](https://arcprize.org/policy).

### Leaderboard Breakdown

| AI System | Author | System Type | ARC-AGI-1 | ARC-AGI-2 | Cost/Task | Code / Paper |
| --- | --- | --- | --- | --- | --- | --- |
| Human Panel | Human | N/A | 98.0% | 100.0% | $17.00 | — |
| GPT-5.2 Pro (High) | OpenAI | CoT | 85.7% | 54.2% | $15.72 | — |
| Gemini 3 Pro (Refine.) | Poetiq | Refinement | N/A | 54.0% | $30.57 | — |
| GPT-5.2 (X-High) | OpenAI | CoT | 86.2% | 52.9% | $1.90 | — |
| Gemini 3 Deep Think (Preview) ² | Google | CoT | 87.5% | 45.1% | $77.16 | — |
| GPT-5.2 (High) | OpenAI | CoT | 78.7% | 43.3% | $1.39 | — |
| GPT-5.2 Pro (Medium) | OpenAI | CoT | 81.2% | 38.5% | $8.99 | — |
| Opus 4.5 (Thinking, 64K) | Anthropic | CoT | 80.0% | 37.6% | $2.40 | — |
| Gemini 3 Flash Preview (High) | Google | CoT | 84.7% | 33.6% | $0.231 | — |
| Gemini 3 Pro | Google | CoT | 75.0% | 31.1% | $0.811 | — |
| Opus 4.5 (Thinking, 32K) | Anthropic | CoT | 75.8% | 30.6% | $1.29 | — |
| Grok 4 (Refine.) | J. Berman | Refinement | 79.6% | 29.4% | $30.40 | [💻](https://github.com/jerber/arc-lang-public) |
| NVARC | ARC Prize 2025 | Custom | N/A | 27.6% | $0.200 | [📄](https://drive.google.com/file/d/1vkEluaaJTzaZiJL69TkZovJUkPSDH5Xc/view) [💻](https://www.kaggle.com/code/gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2/edit?fromFork=1) |
| GPT-5.2 (Medium) | OpenAI | CoT | 72.7% | 26.7% | $0.759 | — |
| Grok 4 (Refine.) | E. Pang | Refinement | 77.1% | 26.0% | $3.97 | [💻](https://github.com/epang080516/arc_agi) |
| Opus 4.5 (Thinking, 16K) | Anthropic | CoT | 72.0% | 22.8% | $0.790 | — |
| GPT-5 Pro | OpenAI | CoT | 70.2% | 18.3% | $7.14 | [📄](https://platform.openai.com/docs/models/gpt-5-pro) |
| GPT-5.1 (Thinking, High) | OpenAI | CoT | 72.8% | 17.6% | $1.17 | — |
| Grok 4 (Thinking) | xAI | CoT | 66.7% | 16.0% | $2.17 | [📄](https://x.ai/news) |
| Opus 4.5 (Thinking, 8K) | Anthropic | CoT | 58.7% | 13.9% | $0.480 | — |
| Claude Sonnet 4.5 (Thinking 32K) | Anthropic | CoT | 63.7% | 13.6% | $0.759 | [📄](https://www.anthropic.com/claude/sonnet) |
| Gemini 3 Flash Preview (Medium) | Google | CoT | 57.7% | 12.8% | $0.082 | — |
| GPT-5 (High) | OpenAI | CoT | 65.7% | 9.9% | $0.730 | — |
| GPT-5.2 (Low) | OpenAI | CoT | 55.7% | 9.7% | $0.264 | — |
| Opus 4.5 (Thinking, 1K) | Anthropic | CoT | 35.2% | 9.4% | $0.230 | — |
| Claude Opus 4 (Thinking 16K) | Anthropic | CoT | 35.7% | 8.6% | $1.93 | [📄](https://www.anthropic.com/news/claude-4) |
| Opus 4.5 (Thinking, None) | Anthropic | Base LLM | 40.0% | 7.8% | $0.220 | — |
| GPT-5 (Medium) | OpenAI | CoT | 56.2% | 7.5% | $0.449 | — |
| Claude Sonnet 4.5 (Thinking 8K) | Anthropic | CoT | 46.5% | 6.9% | $0.235 | [📄](https://www.anthropic.com/claude/sonnet) |
| Claude Sonnet 4.5 (Thinking 16K) | Anthropic | CoT | 48.3% | 6.9% | $0.350 | [📄](https://www.anthropic.com/claude/sonnet) |
| o3 (High) | OpenAI | CoT | 60.8% | 6.5% | $0.834 | — |
| GPT-5.1 (Thinking, Medium) | OpenAI | CoT | 57.7% | 6.5% | $0.421 | — |
| Tiny Recursion Model (TRM) | Bespoke | N/A | 40.0% | 6.3% | $2.10 | [📄](https://alexiajm.github.io/2025/09/29/tiny_recursive_models.html) [💻](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) |
| o4-mini (High) | OpenAI | CoT | 58.7% | 6.1% | $0.856 | — |
| Claude Sonnet 4 (Thinking 16K) | Anthropic | CoT | 40.0% | 5.9% | $0.486 | [📄](https://www.anthropic.com/news/claude-4) |
| Claude Sonnet 4.5 (Thinking 1K) | Anthropic | CoT | 31.0% | 5.8% | $0.142 | [📄](https://www.anthropic.com/claude/sonnet) |
| Grok 4 (Fast Reasoning) | xAI | CoT | 48.5% | 5.3% | $0.061 | [📄](https://x.ai/news/grok-4-fast) |
| o3-Pro (High) | OpenAI | CoT | 59.3% | 4.9% | $7.55 | [📄](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) |
| Gemini 2.5 Pro (Thinking 32K) | Google | CoT | 37.0% | 4.9% | $0.757 | [📄](https://cloud.google.com/blog/products/ai-machine-learning/gemini-2-5-flash-lite-flash-pro-ga-vertex-ai) |
| Claude Opus 4 (Thinking 8K) | Anthropic | CoT | 30.7% | 4.5% | $1.16 | [📄](https://www.anthropic.com/news/claude-4) |
| GPT-5 Mini (High) | OpenAI | CoT | 54.3% | 4.4% | $0.198 | — |
| Gemini 2.5 Pro (Thinking 16K) | Google | CoT | 41.0% | 4.0% | $0.715 | [📄](https://cloud.google.com/blog/products/ai-machine-learning/gemini-2-5-flash-lite-flash-pro-ga-vertex-ai) |
| GPT-5 Mini (Medium) | OpenAI | CoT | 37.3% | 4.0% | $0.063 | — |
| Claude Haiku 4.5 (Thinking 32K) | Anthropic | CoT | 47.7% | 4.0% | $0.377 | [📄](https://www.anthropic.com/news/claude-haiku-4-5) |
| o3 (Preview, Low) ¹ | OpenAI | CoT | 75.7% | 4.0% | $200.00 | [📄](https://arcprize.org/blog/oai-o3-pub-breakthrough) |
| Gemini 2.5 Pro (Preview) | Google | Base LLM | 33.0% | 3.8% | $0.813 | [📄](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) |
| Claude Sonnet 4.5 | Anthropic | Base LLM | 25.5% | 3.8% | $0.130 | [📄](https://www.anthropic.com/claude/sonnet) |
| Gemini 2.5 Pro (Preview, Thinking 1K) | Google | CoT | 31.3% | 3.4% | $0.804 | [📄](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) |
| Gemini 3 Flash Preview (Minimal) | Google | CoT | 21.5% | 3.3% | $0.021 | — |
| o3-mini (High) | OpenAI | CoT | 34.5% | 3.0% | $0.547 | — |
| o3 (Medium) | OpenAI | CoT | 53.8% | 3.0% | $0.479 | — |
| Gemini 2.5 Pro (Thinking 8K) | Google | CoT | 29.5% | 2.9% | $0.444 | [📄](https://cloud.google.com/blog/products/ai-machine-learning/gemini-2-5-flash-lite-flash-pro-ga-vertex-ai) |
| Claude Haiku 4.5 (Thinking 16K) | Anthropic | CoT | 37.3% | 2.8% | $0.139 | [📄](https://www.anthropic.com/news/claude-haiku-4-5) |
| GPT-5 Nano (High) | OpenAI | CoT | 16.7% | 2.6% | $0.029 | — |
| Gemini 2.5 Flash (Preview) (Thinking 24K) | Google | CoT | 32.3% | 2.5% | $0.319 | [📄](https://deepmind.google/models/gemini/flash/) [💻](https://github.com/arcprize/model_baseline) |
| ARChitects | ARC Prize 2024 | Custom | 56.0% | 2.5% | $0.200 | [📄](https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf) [💻](https://www.kaggle.com/code/gregkamradt/arc-prize-v8?scriptVersionId=211457842) |
| o4-mini (Medium) | OpenAI | CoT | 41.8% | 2.4% | $0.231 | — |
| Gemini 2.5 Flash (Preview) (Thinking 1K) | Google | CoT | 16.0% | 2.2% | $0.030 | [📄](https://deepmind.google/models/gemini/flash/) [💻](https://github.com/arcprize/model_baseline) |
| Gemini 2.5 Flash (Preview) (Thinking 8K) | Google | CoT | 25.8% | 2.1% | $0.199 | [📄](https://deepmind.google/models/gemini/flash/) [💻](https://github.com/arcprize/model_baseline) |
| Claude Sonnet 4 (Thinking 8K) | Anthropic | CoT | 29.0% | 2.1% | $0.265 | [📄](https://www.anthropic.com/news/claude-4) |
| o3-mini (Medium) | OpenAI | CoT | 22.3% | 2.1% | $0.284 | — |
| o3-Pro (Low) | OpenAI | CoT | 44.3% | 2.1% | $2.23 | [📄](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) |
| Hierarchical Reasoning Model (HRM) | Bespoke | N/A | 32.0% | 2.0% | $1.68 | — |
| o3 (Low) | OpenAI | CoT | 41.5% | 2.0% | $0.234 | — |
| Gemini 2.5 Flash (Preview) (Thinking 16K) | Google | CoT | 33.3% | 2.0% | $0.317 | [📄](https://deepmind.google/models/gemini/flash/) [💻](https://github.com/arcprize/model_baseline) |
| o3-Pro (Medium) | OpenAI | CoT | 57.0% | 1.9% | $4.74 | [📄](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) |
| GPT-5 (Low) | OpenAI | CoT | 44.0% | 1.9% | $0.190 | — |
| GPT-5.1 (Thinking, Low) | OpenAI | CoT | 33.2% | 1.9% | $0.129 | — |
| Gemini 2.5 Flash (Preview) | Google | CoT | 33.3% | 1.7% | $0.057 | [📄](https://deepmind.google/models/gemini/flash/) [💻](https://github.com/arcprize/model_baseline) |
| o4-mini (Low) | OpenAI | CoT | 21.3% | 1.7% | $0.050 | — |
| GPT-5 Mini (Minimal) | OpenAI | CoT | 5.3% | 1.7% | $0.009 | — |
| Claude Haiku 4.5 (Thinking 8K) | Anthropic | CoT | 25.5% | 1.7% | $0.091 | [📄](https://www.anthropic.com/news/claude-haiku-4-5) |
| Icecuber | ARC Prize 2024 | Custom | 17.0% | 1.6% | $0.130 | [💻](https://www.kaggle.com/code/hansuelijud/template-arc2020-1st-place-solution-by-icecuber) |
| Gemini 2.0 Flash | Google | Base LLM | N/A | 1.3% | $0.004 | [💻](https://github.com/arcprize/model_baseline) |
| Deepseek R1 | Deepseek | CoT | 15.8% | 1.3% | $0.080 | [💻](https://github.com/arcprize/model_baseline) |
| Codex Mini (Latest) | OpenAI | CoT | 27.3% | 1.3% | $0.230 | [📄](https://platform.openai.com/docs/models/codex-mini-latest) |
| Claude Sonnet 4 | Anthropic | Base LLM | 23.8% | 1.3% | $0.127 | [📄](https://www.anthropic.com/news/claude-4) |
| Claude Opus 4 | Anthropic | CoT | 22.5% | 1.3% | $0.639 | [📄](https://www.anthropic.com/news/claude-4) |
| Qwen3-235b-a22b Instruct (25/07) | Alibaba | Base LLM | 11.0% | 1.3% | $0.004 | [📄](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Claude Haiku 4.5 | Anthropic | Base LLM | 14.3% | 1.3% | $0.043 | [📄](https://www.anthropic.com/news/claude-haiku-4-5) |
| Claude Haiku 4.5 (Thinking 1K) | Anthropic | CoT | 16.8% | 1.3% | $0.047 | [📄](https://www.anthropic.com/news/claude-haiku-4-5) |
| Gemini 3 Flash Preview (Low) | Google | CoT | 29.0% | 1.3% | $0.025 | — |
| Deepseek R1 (05/28) | Deepseek | CoT | 21.2% | 1.1% | $0.053 | [📄](https://api-docs.deepseek.com/news/news250528) |
| Claude 3.7 (8K) | Anthropic | CoT | 21.2% | 0.9% | $0.360 | [💻](https://github.com/arcprize/model_baseline) |
| GPT-5 Nano (Medium) | OpenAI | CoT | 20.7% | 0.9% | $0.014 | — |
| Claude Sonnet 4 (Thinking 1K) | Anthropic | CoT | 28.0% | 0.9% | $0.142 | [📄](https://www.anthropic.com/news/claude-4) |
| o1-mini | OpenAI | CoT | 14.0% | 0.8% | $0.191 | [📄](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/) [💻](https://github.com/arcprize/model_baseline) |
| GPT-5 Mini (Low) | OpenAI | CoT | 26.3% | 0.8% | $0.019 | — |
| GPT-5.2 | OpenAI | Base LLM | 12.3% | 0.8% | $0.082 | — |
| Gemini 1.5 Pro | Google | Base LLM | N/A | 0.8% | $0.040 | [💻](https://github.com/arcprize/model_baseline) |
| GPT-4.5 | OpenAI | Base LLM | 10.3% | 0.8% | $2.10 | [💻](https://github.com/arcprize/model_baseline) |
| Claude 3.7 (16K) | Anthropic | CoT | 28.6% | 0.7% | $0.510 | [💻](https://github.com/arcprize/model_baseline) |
| GPT-4.1 | OpenAI | Base LLM | 5.5% | 0.4% | $0.069 | [📄](https://openai.com/index/gpt-4-1/) [💻](https://github.com/arcprize/model_baseline) |
| Grok 3 Mini (Low) | xAI | CoT | 16.5% | 0.4% | $0.013 | [📄](https://x.ai/api#capabilities) |
| GPT-5.1 (Thinking, None) | OpenAI | Base LLM | 5.8% | 0.4% | $0.058 | — |
| Claude 3.7 (1K) | Anthropic | CoT | 11.6% | 0.4% | $0.140 | [💻](https://github.com/arcprize/model_baseline) |
| Claude 3.7 | Anthropic | Base LLM | 13.6% | 0.0% | $0.120 | [💻](https://github.com/arcprize/model_baseline) |
| GPT-4o | OpenAI | Base LLM | 4.5% | 0.0% | $0.080 | [💻](https://github.com/arcprize/model_baseline) |
| GPT-4o-mini | OpenAI | Base LLM | N/A | 0.0% | $0.010 | [💻](https://github.com/arcprize/model_baseline) |
| Avg. Mturker | Human | N/A | 77.0% | N/A | $3.00 | — |
| Stem Grad | Human | N/A | 98.0% | N/A | $10.00 | — |
| Llama 4 Maverick | Meta | Base LLM | 4.4% | 0.0% | $0.012 | [📄](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/) [💻](https://github.com/arcprize/model_baseline) |
| Llama 4 Scout | Meta | Base LLM | 0.5% | 0.0% | $0.006 | [📄](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/) [💻](https://github.com/arcprize/model_baseline) |
| GPT-4.1-Nano | OpenAI | Base LLM | 0.0% | 0.0% | $0.004 | [📄](https://openai.com/index/gpt-4-1/) [💻](https://github.com/arcprize/model_baseline) |
| GPT-4.1-Mini | OpenAI | Base LLM | 3.5% | 0.0% | $0.014 | [📄](https://openai.com/index/gpt-4-1/) [💻](https://github.com/arcprize/model_baseline) |
| o3-mini (Low) | OpenAI | CoT | 14.5% | 0.0% | $0.062 | — |
| Claude Opus 4 (Thinking 1K) | Anthropic | CoT | 27.0% | 0.0% | $0.750 | [📄](https://www.anthropic.com/news/claude-4) |
| Grok 3 | xAI | Base LLM | 5.5% | 0.0% | $0.142 | [📄](https://x.ai/api#capabilities) |
| Magistral Small | Mistral | Base LLM | 5.0% | 0.0% | $0.049 | [📄](https://mistral.ai/news/magistral) |
| Magistral Medium | Mistral | Base LLM | 5.9% | 0.0% | $0.108 | [📄](https://mistral.ai/news/magistral) |
| Magistral Medium (Thinking) | Mistral | CoT | 6.1% | 0.0% | $0.123 | [📄](https://mistral.ai/news/magistral) |
| Gemini 2.5 Pro (Thinking 1K) | Google | CoT | 16.0% | 0.0% | $0.088 | [📄](https://cloud.google.com/blog/products/ai-machine-learning/gemini-2-5-flash-lite-flash-pro-ga-vertex-ai) |
| GPT-5 (Minimal) | OpenAI | Base LLM | 6.0% | 0.0% | $0.056 | — |
| GPT-5 Nano (Low) | OpenAI | CoT | 4.0% | 0.0% | $0.003 | — |
| GPT-5 Nano (Minimal) | OpenAI | CoT | 1.5% | 0.0% | $0.003 | — |
| GPT-5.2 Pro (X-High) | OpenAI | CoT | 90.5% | N/A | $11.65 | — |

#### Notes

Only systems which required less than $10,000 to run are shown.

For models that were not able to produce full test out puts, remaining tasks were marked as incorrect.

Results marked as "preview" are unofficial and may be based on incomplete testing.

<sup>1</sup> ARC-AGI-2 score estimate based on partial testing results and o1-pro pricing.

<sup>2</sup> Provisional cost estimates based on Gemini 3 Pro pricing. Model to be retested once released.