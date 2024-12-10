# Hume + Anthropic - Voice Computer Use Demo on Replit

https://replit.com/@jerhadf/Hume-Anthropic-Computer-Use 

## Getting started
* Add your ANTHROPIC_API_KEY to the Secrets pane
* Add your HUME_API_KEY and config ID from platform.hume.ai 
* Click Run
* Watch the AI work in the Output pane
* Send commands to it in the Webview

## Usage Tips 
* Run `firefox &` in the shell to pre-open Firefox for the agent - must be done before Running
* For changes to the repo to take effect, run `cd computer_use_demo/evi_chat_component/frontend` and `npm run build` in the shell 

## Caution

Computer use is a beta feature. Please be aware that computer use poses unique risks that are distinct from standard API features or chat interfaces. These risks are heightened when using computer use to interact with the internet. To minimize risks, consider taking precautions such as:

* Use a dedicated virtual machine or container with minimal privileges to prevent direct system attacks or accidents. (Covered by using Replit)
* Avoid giving the model access to sensitive data, such as account login information, to prevent information theft.
* Limit internet access to an allowlist of domains to reduce exposure to malicious content.
* Ask a human to confirm decisions that may result in meaningful real-world consequences as well as any tasks requiring affirmative consent, such as accepting cookies, executing financial transactions, or agreeing to terms of service.
* In some circumstances, Claude will follow commands found in content even if it conflicts with the user's instructions. For example, instructions on webpages or contained in images may override user instructions or cause Claude to make mistakes. We suggest taking precautions to isolate Claude from sensitive data and actions to avoid risks related to prompt injection.

Finally, please inform end users of relevant risks and obtain their consent prior to enabling computer use in your own products.

## Credits
Based on https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo
