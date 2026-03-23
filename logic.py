# Comment about border logic

        # Fixing the return type annotation for the function
        def my_function() -> tuple[list[MatchResult], list[MatchResult], list[MatchResult], list[str]]:
            # some logic here
            return value1, value2, value3, value4

        # Adjusting return to meet expectations
        def verify_batch() -> tuple[list[MatchResult], list[MatchResult], list[MatchResult], list[str]]:
            return results_list, results_list, results_list, string_list
