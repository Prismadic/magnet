import re, os, torch, random
import boto3
from spacy.lang.en import English

def _f(
    tag: str = None,
    body: any = None,
    no_print: bool = False,
    luxe: bool = False
):
    """
    The `_f` function is a logging utility that prints messages with different tags and colors based on
    the provided parameters.

    :param tag: The `tag` parameter is a string that represents the tag for the log message. It can be
    one of the following values: "FATAL", "WARN", "INFO", "WAIT", or "SUCCESS"
    :type tag: str
    :param body: The `body` parameter is used to specify the message or content that you want to
    display. It can be of any type
    :type body: any
    :param no_print: The `no_print` parameter is a boolean flag that determines whether the output
    should be printed or returned as a string.
    the formatted string without printing it. If `no_print` is set to `False` (default)
    :type no_print: bool (optional)
    :param luxe: The `luxe` parameter is a boolean flag that determines whether to use a more luxurious
    and colorful output format. If `luxe` is set to `True`, the output will include random colors,
    emojis, and matrix-like characters.
    :type luxe: bool (optional)
    :return: The function `_f` returns a formatted string if the `no_print` parameter is set to `True`.
    If `no_print` is `False`, the function prints the formatted string and returns `None`.
    """
    tags = [
        ("FATAL", "☠️", "\033[91m"),  # Red color for FATAL
        ("WARN", "🚨", "\033[93m"),  # Yellow color for WARN
        ("INFO", "ℹ️", "\033[94m"),  # Blue color for INFO
        ("WAIT", "☕️", "\033[96m"),  # Cyan color for WAIT
        ("SUCCESS", "🌊", "\033[92m"),  # Green color for SUCCESS
    ]
    _luxe = [
        "\033[31m",
        "\033[32m",
        "\033[33m",
        "\033[34m",
        "\033[35m",
        "\033[36m",
        "\033[91m",
        "\033[92m",
        "\033[93m",
        "\033[94m",
        "\033[95m",
        "\033[96m",
    ]
    _matrix = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    _joy = [
        "🍤",
        "🌈",
        "📊",
        "🏁",
        "🌊",
        "🧠",
        "✨",
        "🧮",
        "🎉",
        "🥳",
        "🤩",
        "🐈",
        "❤️",
        "💙",
        "💜",
        "💚",
        "💛",
        "🧡",
        "⭐️",
    ]
    matching_tags = [x for x in tags if x[0] == tag.upper()]
    if matching_tags:
        tag_text = matching_tags[0][0]
        emoji = matching_tags[0][1]
        color_code = matching_tags[0][2]
        if luxe:
            return (
                f"{_luxe[random.randint(0,len(_luxe)-1)]} {_joy[random.randint(0,len(_joy)-1)]} {_matrix[random.randint(0,len(_matrix)-1)]}: {body}\033[0m"
                if no_print
                else print(
                    f"{_luxe[random.randint(0,len(_luxe)-1)]} {_joy[random.randint(0,len(_joy)-1)]} {_matrix[random.randint(0,len(_matrix)-1)]}: {body}\033[0m"
                )
            )
        else:
            return (
                f"{color_code} {emoji} {tag_text}: {body}\033[0m"
                if no_print
                else print(f"{color_code}{emoji} {tag_text}: {body}\033[0m")
            )
    else:
        print(f"😭 UNKNOWN TAG - `{tag}`")

class Utils:
    def __init__(self):
        nlp = English()
        nlp.add_pipe("sentencizer")
        self.nlp = nlp

    def check_cuda(self):
        """
        The function checks if CUDA is available on the machine and provides additional information if
        it is.
        """
        if torch.cuda.is_available():
            _f("success", "CUDA is available on this machine.")
            # You can also print additional information like the number of available GPUs:
            _f("info", f"Number of available GPUs - {torch.cuda.device_count()}")
            # To get the name of the GPU:
            _f(
                "info", f"GPU Name - {torch.cuda.get_device_name(0)}"
            )  # 0 is the GPU index
            return True
        else:
            _f("warn", "CUDA is not available on this machine.")

    def normalize_text(self, _):
        """
        The `clean` function takes a string as input and performs various cleaning operations on it,
        such as removing whitespace, replacing characters, and removing curly braces.

        :param _: The parameter "_" is a placeholder for the input string that needs to be cleaned
        :return: a cleaned version of the input string.
        """
        if isinstance(_, list):
            _f(
                "warn",
                f"this item may not process properly because it is a list: \n{_}",
            )
        try:
            if not isinstance(_, str):
                _f('warn', f'non-string found {type(_)}')
            _ = _.strip()
            _ = _.replace('.', '') if (_.count('.') / len(_)) * 100 > 0.3 else _

            # Check if more than 20% of the string is integers
            num_digits = sum(1 for char in _ if char.isdigit())
            if (num_digits / len(_)) >= 0.2:
                _ = ""  # Replace with empty quotes
            else:
                _ = _.replace('"', "'")

            # Check if the string has at least 5 words
            words = _.split()
            if len(words) < 5:
                _ = ""  # Replace with empty quotes if fewer than 5 words

            _ = _.replace("\t", "")
            _ = _.replace("\n", "")
            _ = _.replace("\xa0", "")
            _ = " ".join(_.split())
            _ = re.sub(r" {[^}]*}", "", _)
            return str(_)
        except Exception as e:
            _f("fatal", e)


    def upload_to_s3(
        self,
        file_or_dir: str = None,
        keys: tuple = ("YOUR_ACCESS_KEY", "YOUR_SECRET_KEY"),
        bucket: str = None,
        bucket_path: str = None,
    ):
        """
        The function `upload_to_s3` uploads a file to an S3 bucket using the provided access keys and
        bucket information.

        :param file: The `file` parameter is the path to the file that you want to upload to Amazon S3.
        It should be a string representing the file's location on your local machine
        :param keys: The `keys` parameter is a tuple that contains the AWS access key and secret access
        key. These keys are used to authenticate and authorize access to the AWS S3 service
        :type keys: tuple
        :param bucket: The `bucket` parameter is the name of the S3 bucket where you want to upload the
        file
        :type bucket: str
        :param bucket_path: The `bucket_path` parameter is the path within the S3 bucket where you want
        to upload the file. It is a string that represents the directory structure within the bucket.
        For example, if you want to upload the file to the root of the bucket, you can set `bucket_path`
        to
        :type bucket_path: str
        """
        aws_access_key_id, aws_secret_access_key = keys

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        _f("warn", f"uploading to S3 - {file_or_dir}")
        if os.path.isfile(os.path.abspath(file_or_dir)):
            s3.upload_file(
                os.path.abspath(file_or_dir), bucket, f'{bucket_path}/{file_or_dir.split("/")[-1]}'
            )
            _f("success", f'uploaded - {bucket}/{bucket_path}/{file_or_dir.split("/")[-1]}')
        elif os.path.isdir(os.path.abspath(file_or_dir)):
            for filename in os.listdir(file_or_dir):
                f = os.path.join(file_or_dir, filename)
                s3.upload_file(
                    os.path.abspath(f), bucket, f'{bucket_path}/{f.split("/")[-1]}'
                )
                _f("success", f'uploaded - {bucket}/{bucket_path}/{f.split("/")[-1]}')
            
        
        
