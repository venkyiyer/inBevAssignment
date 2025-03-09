## Introduction
The authors in this paper are conveying the message that CoT reasoning approach is better suited when compared to prompting for an LLM while solving complex problems.
The authors also mention that the LLM's have intrinsic capability to eason without prompting at all. This reaasoning capability of an LLM can be enhance by teaching them to reason through CoT reasoning. The authors have highlighted their findings in figure. 1 that without prompting and allowing the LLM to reason across top-k tokens, the LLM is able to decode the final answer in a effectie way. 

## Contributions made by the authors
Simple problems are intrinsically solved by the LLM without the use of prompting
Explore the intrinsic capabilities of the LLM without any human intervention or direction (prompts)
Suggest the usage of CoT decoding for better reasoning
If the LLM's are allowed to reason over top-k tokens, it is seen that the model naturally generated CoT paths

## Experiment observations
The experiments were performed on LLM's like Gemma-7B, Mistral-7B and PaLM-2
It is observed that there has been an increase of accuracy in terms of reasoning with CoT based decoding approach
The choice of k also influences the models reasoning capability
Table 6 shows an comparison between Greedy decoding and CoT decoding 

## Additional observation
The authors discuss the combination of CoT decoding with CoT prompting to yield even better resoning results (Table 7 shows the comparison)

## Conclusion
The CoT decoding approach is introduced to extract more reliable decoding paths for LLM's



