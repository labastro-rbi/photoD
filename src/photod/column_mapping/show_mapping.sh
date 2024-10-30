:
# Creation of sample column mapping from sample data
#
# Add the argument --results=json to get something easily parseable

echo "Columns that can be automatically mapped to parameters"
arq \
    --query=find_mapped.rq \
    --data=column_mapping.ttl \
    --data=sample_function.ttl \
    --data=sample_table.ttl \
    --data=sample_glossary.ttl
echo "Parameters that do not have an exact matching concept in the columns"
arq \
    --query=find_unmapped.rq \
    --data=column_mapping.ttl \
    --data=sample_function.ttl \
    --data=sample_table.ttl \
    --data=sample_glossary.ttl
