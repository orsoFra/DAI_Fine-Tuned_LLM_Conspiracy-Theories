import os

class Prompter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def simple_prompt(input_doc:str, is_formatted: bool):
        format = 'Use the following JSON format for your output: {{"Answer": "[your answer here]"}}' if is_formatted else ''
        prompt = f'Decide whether the following transcription of a video talks about a conspiracy theory or not (if yes output = 1/else  output=0). Provide just your output, no justification.{format} .“{input_doc}"'
        return prompt
    
    @staticmethod
    def definition_prompt(input_doc:str, is_formatted: bool):
        format = 'Use the following JSON format for yout output: {{"Answer": "[your answer here]", "Justification" : "[your justification here]"}}' if is_formatted else ''
        prompt = f'Given this definition of conspiracy theory: A conspiracy theory is a belief that two or more actors have coordinated in secret to achieve an outcome and that their conspiracy is of public interest but not public knowledge. Conspiracy theories are oppositional, which means they oppose publicly accepted understandings of events; describe malevolent or forbidden acts;  ascribe agency to individuals and groups rather than to impersonal or systemic forces;  are epistemically risky, meaning that though they are not necessarily false or implausible, taken collectively they are more prone to falsity than other types of belief; and are social constructs that are not merely adopted by individuals but are shared with social objectives in mind, and they have the potential not only to represent and interpret reality but also to fashion new social realities .Decide whether the following transcription of a video talks about a conspiracy theory or not (if yes output = 1/else  output=0). {format}.“{input_doc}"'
        return prompt
    
    @staticmethod
    def sbs_prompt(input_doc:str, is_formatted: bool):
        prompt = f'Decide whether the following transcription of a video talks about a conspiracy theory or not (if yes output = 1/else output=0). First, extract the narrative or claim from the text. Second, decide if the claim talks about a conspiracy theory. Third, answer the question (if yes output = 1/else  output=0). “{input_doc}"'
        return prompt
    
    @staticmethod
    def sp_with_context(input_doc:str, is_formatted: bool, context: str):
        prompt = Prompter.simple_prompt(input_doc=input_doc,is_formatted=is_formatted)
        if(len(context)>4):
            prompt += f'Take in account the following description of the video to make your decision: "{context}"'
        return prompt
    
    @staticmethod
    def sbs_with_context(input_doc:str, context: str, is_formatted: bool):
        prompt = Prompter.sbs_prompt(input_doc=input_doc,is_formatted=is_formatted)
        if(len(context)>4):
            prompt += f'Take in account the following description of the video to make your decision: "{context}"'
        return prompt

    @staticmethod
    def dp_with_context(input_doc:str, is_formatted: bool, context: str):
        prompt = Prompter.definition_prompt(input_doc=input_doc,is_formatted=is_formatted)
        if(len(context)>4):
            prompt += f'Take in account the following description of the video to make your decision: "{context}"'
        return prompt

    def print_message(content):
        return [{
        'role': 'user',
        'content': str(content),#item[1] is the VTT
        }]
    
    @staticmethod
    def simple_prompt_think(input_doc:str, is_formatted: bool):
        format = 'Use the following JSON format for your output: {{"Answer": "[your answer here]"}}' if is_formatted else ''
        prompt = f'Decide whether the following transcription of a video talks about a conspiracy theory or exhibits signs of conspiratorial thinking  (if yes output = 1/else  output=0). Provide just your output, no justification.{format} .“{input_doc}"'
        return prompt
    
    @staticmethod
    def definition_prompt_think(input_doc:str, is_formatted: bool):
        format = 'Use the following JSON format for yout output: {{"Answer": "[your answer here]", "Justification" : "[your justification here]"}}' if is_formatted else ''
        prompt = f'Given this definition of conspiracy theory: A conspiracy theory is a belief that two or more actors have coordinated in secret to achieve an outcome and that their conspiracy is of public interest but not public knowledge. Conspiracy theories are oppositional, which means they oppose publicly accepted understandings of events; describe malevolent or forbidden acts;  ascribe agency to individuals and groups rather than to impersonal or systemic forces;  are epistemically risky, meaning that though they are not necessarily false or implausible, taken collectively they are more prone to falsity than other types of belief; and are social constructs that are not merely adopted by individuals but are shared with social objectives in mind, and they have the potential not only to represent and interpret reality but also to fashion new social realities .Decide whether the following transcription of a video talks about a conspiracy theory or exhibits signs of conspiratorial thinking  (if yes output = 1/else  output=0). {format}.“{input_doc}"'
        return prompt
    
    @staticmethod
    def sbs_prompt_think(input_doc:str, is_formatted: bool):
        prompt = f'Decide whether the following transcription of a video talks about a conspiracy theory or not (if yes output = 1/else output=0). First, extract the narrative or claim from the text. Second, decide if the claim talks about a conspiracy theory or exhibits signs of conspiratorial thinking. Third, answer the question (if yes output = 1/else  output=0). “{input_doc}"'
        return prompt

    @staticmethod
    def sp_with_context_think(input_doc:str, is_formatted: bool, context: str):
        prompt = Prompter.simple_prompt_think(input_doc=input_doc,is_formatted=is_formatted)
        if(len(context)>4):
            prompt += f'Take in account the following description of the video to make your decision: "{context}"'
        return prompt
    
    @staticmethod
    def sbs_with_context_think(input_doc:str, context: str, is_formatted: bool):
        prompt = Prompter.sbs_prompt_think(input_doc=input_doc,is_formatted=is_formatted)
        if(len(context)>4):
            prompt += f'Take in account the following description of the video to make your decision: "{context}"'
        return prompt

    @staticmethod
    def dp_with_context_think(input_doc:str, is_formatted: bool, context: str):
        prompt = Prompter.definition_prompt_think(input_doc=input_doc,is_formatted=is_formatted)
        if(len(context)>4):
            prompt += f'Take in account the following description of the video to make your decision: "{context}"'
        return prompt
    
    @staticmethod
    def simple_prompt_end(input_doc:str, is_formatted: bool, context: str):
        format = 'Use the following JSON format for your output: {{"Answer": "[your answer here]"}}' if is_formatted else ''
        prompt = f'Transcription:“{input_doc}" . Decide whether the given transcription of a video talks about a conspiracy theory or not (if yes output = 1/else  output=0). Provide just your output, no justification.{format} .'
        return prompt
    
    @staticmethod
    def definition_prompt_end(input_doc:str, is_formatted: bool, context: str):
        format = 'Use the following JSON format for yout output: {{"Answer": "[your answer here]", "Justification" : "[your justification here]"}}' if is_formatted else ''
        prompt = f'Transcription:“{input_doc}".Given this definition of conspiracy theory: A conspiracy theory is a belief that two or more actors have coordinated in secret to achieve an outcome and that their conspiracy is of public interest but not public knowledge. Conspiracy theories are oppositional, which means they oppose publicly accepted understandings of events; describe malevolent or forbidden acts;  ascribe agency to individuals and groups rather than to impersonal or systemic forces;  are epistemically risky, meaning that though they are not necessarily false or implausible, taken collectively they are more prone to falsity than other types of belief; and are social constructs that are not merely adopted by individuals but are shared with social objectives in mind, and they have the potential not only to represent and interpret reality but also to fashion new social realities .Decide whether the given transcription of a video talks about a conspiracy theory or not (if yes output = 1/else  output=0). {format}'
        return prompt
    
    @staticmethod
    def sbs_prompt_end(input_doc:str, is_formatted: bool, context: str):
        prompt = f'Transcription:“{input_doc}". Decide whether the given transcription of a video talks about a conspiracy theory or not (if yes output = 1/else output=0). First, extract the narrative or claim from the text. Second, decide if the claim talks about a conspiracy theory. Third, answer the question (if yes output = 1/else  output=0).'
        return prompt

    def simple_prompt_end_video(video_path: str, file_name: str, fps: int = 1, is_formatted: bool = False):
        full_path = os.path.join(video_path, file_name)

        format_hint = (
            'Use the following JSON format for your output: {{"Answer": "[your answer here]"}}.'
            if is_formatted else ''
        )

        text_prompt = (
            "Decide whether the provided video talks about a conspiracy theory or not (if yes output = 1/else  output=0). Provide just your output, no justification. {format}"
        )

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
             	"role": "user",
                "content": [
                    {
                     	"type": "video",
                        "video": {
                            "video_path": full_path,
                            "fps": fps
                        }
                    },
                    {
                     	"type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]
        return conversation

    def definition_prompt_end_video(video_path: str, file_name: str, fps: int = 1, is_formatted: bool = False):
        full_path = os.path.join(video_path, file_name)

        format_hint = (
            'Use the following JSON format for your output: {"Answer": "[your answer here]", "Justification" : "[your justification here]"}'
            if is_formatted else ''
        )

        text_prompt = (
            "Given this definition of conspiracy theory: A conspiracy theory is a belief that two or more actors have coordinated in secret to " +
            "achieve an outcome and that their conspiracy is of public interest but not public knowledge. Conspiracy theories are oppositional, " +
            "which means they oppose publicly accepted understandings of events; describe malevolent or forbidden acts; " +
            "ascribe agency to individuals and groups rather than to impersonal or systemic forces;  are epistemically risky, " +
            "meaning that though they are not necessarily false or implausible, taken collectively they are more prone to falsity than other types of belief; " +
            "and are social constructs that are not merely adopted by individuals but are shared with social objectives in mind, " +
            "and they have the potential not only to represent and interpret reality but also to fashion new social realities. " +
            "Decide whether the given transcription of a video talks about a conspiracy theory or not (if yes output = 1/else  output=0). " +
            f"{format_hint}"
        )

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": full_path,
                            "fps": fps
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]

        return conversation

    def sbs_prompt_end_video(video_path: str, file_name: str, fps: int = 1, is_formatted: bool = False):
        full_path = os.path.join(video_path, file_name)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": full_path,
                            "fps": fps
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Decide whether the provided video is related to a conspiracy theory or not " +
                            "(if yes output = 1/else output=0). " +
                            "First, extract the narrative or claim. " +
                            "Second, decide if the claim is related to a conspiracy theory. " +
                            "Third, answer the question (if yes output = 1/else output=0)."
                        )
                    }
                ]
            }
        ]

        return conversation
