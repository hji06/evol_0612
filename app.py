let
    // 원본 테이블 이름만 맞춰줘
    Source = Excel.CurrentWorkbook(){[Name="subject_list2"]}[Content],

    // 1) strㄴ
    OnlyStr = Table.SelectRows(Source, each [분반타입] = "str"),

    // 2) 분반 텍스트 정리 + 분반그룹(첫 글자)
    CleanClass = Table.TransformColumns(
        OnlyStr,
        {
            {"분반", each Text.Upper(Text.Trim(Text.From(_))), type text},
            {"과목명", each Text.Trim(Text.From(_)), type text},
            {"담당교사", each if _=null then "" else Text.Trim(Text.From(_)), type text},
            {"수업반", each if _=null then "" else Text.Trim(Text.From(_)), type text}
        }
    ),
    AddGroup = Table.AddColumn(CleanClass, "분반그룹", each Text.Start([분반], 1), type text),

    // 3) A/B/C/D로 시작하는 분반만 (A1 같은 것도 포함)
    OnlyABCD = Table.SelectRows(AddGroup, each List.Contains({"A","B","C","D"}, [분반그룹])),

    // 4) (분반): 수업반 + 줄바꿈 + 담당교사  => 세트 문자열 만들기
    AddSetText = Table.AddColumn(
        OnlyABCD,
        "세트",
        each
            "(" & [분반] & "): " & [수업반] &
            "#(lf)" & [담당교사],
        type text
    ),

    // 5) (분반그룹, 과목명) 기준으로 세트들 모아서 "줄바꿈 2번"으로 합치기
    Grouped = Table.Group(
        AddSetText,
        {"분반그룹","과목명"},
        {{"셀값", each Text.Combine(List.Distinct([세트]), "#(lf)#(lf)"), type text}}
    ),

    // 6) X축=과목명으로 Pivot, Y축=분반그룹
    Pivoted = Table.Pivot(
        Grouped,
        List.Sort(List.Distinct(Grouped[과목명])),
        "과목명",
        "셀값",
        each Text.Combine(List.Distinct(_), "#(lf)#(lf)")
    ),

    // 7) 분반그룹 정렬(A,B,C,D)
    AddSortKey = Table.AddColumn(Pivoted, "_sort", each List.PositionOf({"A","B","C","D"}, [분반그룹]), Int64.Type),
    Sorted = Table.Sort(AddSortKey, {{"_sort", Order.Ascending}}),
    Result = Table.RemoveColumns(Sorted, {"_sort"})
in
    Result
