import time


class SpeedTimer():
    def __init__(self):
        self._start_time = 0.
        self._end_time = 0.
        self.times = 0.
        self.num = 0

    def __enter__(self):
        self._start_time = time.time()
        # return self #with ~ as ~:を使う際は必要

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()
        self.times += self._end_time - self._start_time
        self.num += 1

    def __call__(self, reset=True):
        # time = self.times / max(1, self.num)
        total_time = self.times
        avg_time = self.times / self.num if self.num > 0 else 0.
        num_calls = self.num
        if reset:
            self._reset()
        return {"total_time": total_time, "avg_time": avg_time, "num_calls": num_calls}

    def _reset(self):
        self._start_time = 0.
        self._end_time = 0.
        self.times = 0.
        self.num = 0


class SpeedTester():
    def __init__(self):
        self._timers = {}

    def __getitem__(self, key):
        if not key in self._timers:
            self._timers[key] = SpeedTimer()
        return self._timers[key]  # キーに対応する値を取得

    def __setitem__(self, key, value):
        raise NotImplementedError("This class does not support item assignment")

    def __delitem__(self, key):
        del self._timers[key]  # キーに対応する項目を削除

    def items(self):
        return self._timers.items()

    def __str__(self):
        results = []
        for key, timer in self._timers.items():
            stats = timer()
            results.append(
                f"{key}:\n"
                f"├─Avg Time: {self.format_with_si_prefix(stats['avg_time'])}s\n"
                f"├─Total Time: {self.format_with_si_prefix(stats['total_time'])}s\n"
                f"└─Number of Calls: {self.format_with_si_prefix(stats['num_calls'])}"
            )
        return "\n".join(results)

    def decorator(self, key):
        def wrapper(func):
            def wrapped(*args, **kwargs):
                with self[key]:
                    result = func(*args, **kwargs)
                return result
            return wrapped
        return wrapper

    def format_with_si_prefix(self, value):
        # SI接頭語の定義
        si_prefixes = [
            (1e24, 'Y'),  # yotta
            (1e21, 'Z'),  # zetta
            (1e18, 'E'),  # exa
            (1e15, 'P'),  # peta
            (1e12, 'T'),  # tera
            (1e9,  'G'),  # giga
            (1e6,  'M'),  # mega
            (1e3,  'k'),  # kilo
            (1e0,  ''),   # (none)
            (1e-3, 'm'),  # milli
            (1e-6, 'µ'),  # micro
            (1e-9, 'n'),  # nano
            (1e-12, 'p'),  # pico
            (1e-15, 'f'),  # femto
            (1e-18, 'a'),  # atto
            (1e-21, 'z'),  # zepto
            (1e-24, 'y'),  # yocto
        ]

        # 適切な接頭語を探す
        for factor, prefix in si_prefixes:
            if value >= factor:
                scaled_value = value / factor
                return f"{scaled_value:.2f} {prefix}"

        # 万が一、全ての条件に当てはまらない場合は、そのままの値を返す
        return str(value)


if __name__ == "__main__":
    Tester = SpeedTester()
    with Tester["pre_process"]:
        """
        pre process
        """
        time.sleep(0.3)
    with Tester["main_process"]:
        """
        main process
        """
        time.sleep(0.2)
    with Tester["pre_process"]:
        """
        pre process
        """
        time.sleep(0.2)

    @Tester.decorator("DECO_pre_process")
    def pre_process():
        # 前処理
        time.sleep(0.3)

    @Tester.decorator("DECO_main_process")
    def main_process():
        # メイン処理
        time.sleep(0.2)

    pre_process()
    main_process()
    pre_process()

    print(Tester)
