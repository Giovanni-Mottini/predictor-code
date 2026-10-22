import io

from shared_code import progress


def test_progress_is_flushed_and_reports_phase_count_and_monotonic_elapsed(monkeypatch):
    class Log(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()

    clock = iter([10.0, 3671.9])
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(clock))
    log = Log()
    reporter = progress.ProgressReporter("Xsens", stream=log)
    reporter.update("recordings processed", completed=1, total=4)
    assert log.getvalue() == "[Xsens] recordings processed | 1/4 (25%) | elapsed 01:01:01\n"
    assert log.flushes == 1


def test_progress_uses_stderr_and_handles_empty_selection(capsys):
    reporter = progress.ProgressReporter("Xsens")
    reporter.update("no selected recordings", completed=0, total=0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "0/0" in captured.err and "%" not in captured.err


def test_disabled_progress_is_silent(capsys):
    reporter = progress.ProgressReporter("Garmin", enabled=False)
    reporter.update("HRV block", completed=1, total=4)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
