# A special gift for Professor Hu

from requests import get
import multiprocessing
import time
import datetime
from tqdm import tqdm


class TAO:
    NUM_OF_PROCESS = 100
    INTERVAL = 120

    @staticmethod
    def click_hzs():
        while "天下难事，必作于易；天下大事，必作于细":
            get(
                "http://www.cs.cqu.edu.cn/system/resource/code/datainput.jsp?owner=1331212424&newsid=4215"
            )
        return "大方无隅，大器晚成，大音希声，大象无形"

    @staticmethod
    def get_hzs():
        while "祸兮，福之所倚；福兮，祸之所伏":
            r = get(
                "http://www.cs.cqu.edu.cn/system/resource/code/news/click/dynclicks.jsp?clickid=4212&owner=1331212424&clicktype=wbnews"
            )

            if r.status_code == 200:
                return int(r.content.decode("utf-8")) / 1000
        return "道，可道，非常道；名，可名，非常名"

    def __init__(self, num_of_process=None, interval=None):
        print(
            """
        TTTTTTTTTTTTTTTTTTTTTTT                                
        T:::::::::::::::::::::T                                
        T:::::::::::::::::::::T                                
        T:::::TT:::::::TT:::::T                                
        TTTTTT  T:::::T  TTTTTTaaaaaaaaaaaaa     ooooooooooo   
                T:::::T        a::::::::::::a  oo:::::::::::oo 
                T:::::T        aaaaaaaaa:::::ao:::::::::::::::o
                T:::::T                 a::::ao:::::ooooo:::::o
                T:::::T          aaaaaaa:::::ao::::o     o::::o
                T:::::T        aa::::::::::::ao::::o     o::::o
                T:::::T       a::::aaaa::::::ao::::o     o::::o
                T:::::T      a::::a    a:::::ao::::o     o::::o
              TT:::::::TT    a::::a    a:::::ao:::::ooooo:::::o
              T:::::::::T    a:::::aaaa::::::ao:::::::::::::::o
              T:::::::::T     a::::::::::aa:::aoo:::::::::::oo 
              TTTTTTTTTTT      aaaaaaaaaa  aaaa  ooooooooooo   \n\n"""
        )

        TAO.NUM_OF_PROCESS = (
            num_of_process if num_of_process is not None else TAO.NUM_OF_PROCESS
        )
        TAO.INTERVAL = interval if interval is not None else TAO.INTERVAL

        # create process_pool
        process_pool = []

        print(
            "\033[1;31;40m[1] Tao gave birth to the One\033[0m Create process_pool, number of process:{}".format(
                TAO.NUM_OF_PROCESS
            )
        )
        for i in tqdm(range(TAO.NUM_OF_PROCESS)):
            process_pool.append(multiprocessing.Process(target=self.click_hzs, args=()))

        # activate process in process_pool
        print(
            "\033[1;31;40m[2] One gave birth successively to two\033[0m Activate process in process_pool"
        )
        for i in tqdm(range(TAO.NUM_OF_PROCESS)):
            process_pool[i].start()

        # get the result
        print(
            "\033[1;31;40m[3] Two gave birth successively to three, then infinity arise\033[0m\n"
            "Congratulations! Tao is running now!! \n"
            "Follows are what we have achieved (The result will refresh every {} seconds):\n".format(
                TAO.INTERVAL
            )
            + "=" * 77
        )
        res = self.get_hzs()
        print(
            "[{}] So far it has had {}k hits".format(
                datetime.datetime.now().strftime("%y-%m-%d %I:%M:%S %p"), round(res, 2),
            )
        )
        while True:
            pre_time = datetime.datetime.now()
            pre_res = res

            time.sleep(TAO.INTERVAL)

            res = self.get_hzs()
            print(
                "[{}] So far it has had {}k hits, which are growing at {}k hits/s".format(
                    datetime.datetime.now().strftime("%y-%m-%d %I:%M:%S %p"),
                    round(res, 2),
                    round(
                        (res - pre_res) / (datetime.datetime.now() - pre_time).seconds,
                        4,
                    ),
                )
            )
