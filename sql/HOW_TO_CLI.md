Step 1) Get your passeord and project-ref

``` bash
# postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres

postgresql://postgres:mamapinga-2025@db.npjfjrxolirvexivxfud.supabase.co:5432/postgres
```

Step 2) 

psql "postgresql://postgres:mamapinga-2025@db.npjfjrxolirvexivxfud.supabase.co:5432/postgres" -f sql/bm25/bm25_trigger.sql


Step 3) Watch it run


Step 4) Verify the trigger in Supabase web client




