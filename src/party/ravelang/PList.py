from src.party.maths import max
from src.party.ravelang.PNode import PNode

class PList(PNode):
    def __init__(self, phrases, prefix='', suffix=''):
        import re
        if isinstance(phrases, str):
            if ';' in phrases or '\n' in phrases or ',' in phrases:
                # phrases split on newlines and semicolons
                phrases = re.split(r'[\n;,]', phrases)
                phrases = [phrase for phrase in phrases if len(phrase.strip()) > 0]
                phrases = [f'1.0 {prefix}{phrase.strip()}{suffix}' for phrase in phrases]
            else:
                # phrases split on spaces
                phrases = phrases.split(' ')
                phrases = [f'1.0 {prefix}{phrase.strip()}{suffix}' for phrase in phrases]

            s = '\n'.join(phrases)
            nodes, max = parse_promptlines(s)
        elif isinstance(phrases, PNode):
            nodes = phrases

        super(PList, self).__init__(nodes)

    # def get_debug_string(self,
    #                      t_length=10,
    #                      t_timestep=0.5):
    #     return f"PList(children={len(self.children)})"

def parse_promptlines(promptstr, prefix='', suffix=''):
    w_max = 0
    ret_nodes = []

    for text in promptstr.splitlines():
        if not text or not text.strip():
            continue

        text = text.strip()

        # [weight] [text]
        parts = text.split(' ')
        weight = float(parts[0])
        text = f"{prefix}{' '.join(parts[1:])}{suffix}"

        # Split the text and interpret each token
        # Example text: "Aerial shot of __________ mountains by Dan Hillier, drawn with psychedelic white ink on black paper"
        tokens = text.split(' ')
        pmt = PNode(text, weight)
        w_max = max(w_max, weight)
        ret_nodes.append(pmt)

    return ret_nodes, w_max
