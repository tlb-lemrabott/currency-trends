{
    "success": true,
    "message": "Successfully retrieved currencies",
    "data": [
        {
            "id": 1,
            "nameFr": "Dollar US",
            "nameAr": "الدولار الأمريكي",
            "unity": 1,
            "code": "USD",
            "exchangeRates": [
                {
                    "id": 137058,
                    "day": "2016-06-14", //today date
                    "value": "333.21", // price of exchange
                    "endDate": "2016-06-15" // last day of validity for this price
                },
                {
                    "id": 173075,
                    "day": "2025-08-20", //today date
                    "value": "389.24", // price of exchange
                    "endDate": "2025-08-21" // last day of validity for this price
                },
                {
                    "id": 173094,
                    "day": "2025-08-21",  //today date
                    "value": "392.29",  // price of exchange
                    "endDate": "2025-08-25"  // last day of validity for this price
                }
            ]
        },
        {
            "id": 2,
            "name_fr": "Euro",
            "nameAr": "يورو",
            "unity": 1,
            "code": "EUR",
            "exchangeRates": [
                {
                    "id": 137052,
                    "day": "2016-06-14", //today date
                    "value": "423.21", // price of exchange
                    "endDate": "2016-06-15" // last day of validity for this price
                },
                {
                    "id": 137071,
                    "day": "2025-08-20", //today date
                    "value": "433.24", // price of exchange
                    "endDate": "2025-08-21" // last day of validity for this price
                },
                {
                    "id": 137128,
                    "day": "2025-08-21",  //today date
                    "value": "440.29",  // price of exchange
                    "endDate": "2025-08-25"  // last day of validity for this price
                }
            ]
        }
    ]
}
I have theis json schema of data represent some currencies exchange prices by a central bank in a country.
The data represent currencies exchange prices from Jun 2016 to August 2025. every two days or three we get a new exchangeRates object added.

How to build script or application or something to analyse these data and get as output the status of curencies from 2016 until 2025 represented in a curve and exctract the possibilities or probabilities of there status in the comming months and years based on the past experience.
Which is more powerful and suitable in doing this, java or python 
?

Note I only want to use completly free tools if possible.