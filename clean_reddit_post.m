function cleaned_text = clean_reddit_post(raw_text)
    % Enhanced cleaning function for Reddit posts
    if isempty(raw_text)
        cleaned_text = '';
        return;
    end
    
    % Convert to string if needed
    if iscell(raw_text)
        raw_text = raw_text{1};
    end
    
    % Remove HTML tags and entities
    cleaned_text = regexprep(raw_text, '<[^>]*>', ' ');
    cleaned_text = regexprep(cleaned_text, '&[a-zA-Z]+;', ' ');
    
    % Remove URLs
    cleaned_text = regexprep(cleaned_text, 'https?://\S+', ' ');
    cleaned_text = regexprep(cleaned_text, 'www\.\S+', ' ');
    
    % Remove Reddit-specific formatting
    cleaned_text = regexprep(cleaned_text, '/u/\w+', ' '); % Remove usernames
    cleaned_text = regexprep(cleaned_text, '/r/\w+', ' '); % Remove subreddit references
    cleaned_text = regexprep(cleaned_text, '\*+', ' '); % Remove asterisks
    cleaned_text = regexprep(cleaned_text, '^#+\s*', ' '); % Remove headers
    
    % Remove excessive whitespace and normalize
    cleaned_text = regexprep(cleaned_text, '\s+', ' ');
    cleaned_text = strtrim(cleaned_text);
    
    % Remove very short or problematic content
    if length(cleaned_text) < 3
        cleaned_text = '';
    end
end
