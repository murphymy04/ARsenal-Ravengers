GET http://localhost:8000/retrieve/Peter%20Nguyen
HTTP 200
[Asserts]
jsonpath "$.person" == "Peter Nguyen"
jsonpath "$.facts" count >= 1
jsonpath "$.prompt" isString
jsonpath "$.prompt" != ""
